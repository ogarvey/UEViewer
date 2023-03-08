#include "Core.h"
#include "UnCore.h"

#include "UnObject.h"
#include "UnrealMaterial/UnMaterial.h"

#include "Mesh/SkeletalMesh.h"
#include "Mesh/StaticMesh.h"

#include "UnrealMesh/UnMathTools.h"

#include "Exporters.h"
#include "../UmodelTool/Version.h"

#define FIRST_BONE_NODE		1

//?? TODO: remove this function
static CVec3 GetMaterialDebugColor(int Index)
{
	// most of this code is targeted to make maximal color combinations
	// which are maximally different for adjacent BoneIndex values
	static const float table[]  = { 0.9f, 0.3f, 0.6f, 1.0f };
	static const int  table2[] = { 0, 1, 2, 4, 7, 3, 5, 6 };
	Index = (Index & 0xFFF8) | table2[Index & 7] ^ 7;
	#define C(x)	( (x) & 1 ) | ( ((x) >> 2) & 2 )
	CVec3 r;
	r[0] = table[C(Index)];
	r[1] = table[C(Index >> 1)];
	r[2] = table[C(Index >> 2)];
	#undef C
	return r;
}

// Fix vector orientation: glTF has right-handed coordinate system with Y up,
// Unreal uses left-handed system with Z up. Also, Unreal uses 'cm' scale,
// while glTF 'm'.

inline void TransformPosition(CVec3& pos)
{
	Exchange(pos[1], pos[2]);
	pos.Scale(0.01f);
}

inline void TransformDirection(CVec3& vec)
{
	Exchange(vec[1], vec[2]);
}

inline void TransformRotation(CQuat& q)
{
	Exchange(q.Y, q.Z);		// same logic as for vector
//??	q.W *= -1;				// changing left-handed to right-handed, so inverse rotation - works correctly without this line
}

struct BufferData
{
	enum
	{
		BYTE = 5120,
		UNSIGNED_BYTE = 5121,
		SHORT = 5122,
		UNSIGNED_SHORT = 5123,
		UNSIGNED_INT = 5125,
		FLOAT = 5126
	};

	enum BufferTargetType
	{
		// unspecified/other - don't emit "target"
		OTHER_BUFFER = 0,
		// contains vertex attributes (position, normals, ...)
		ARRAY_BUFFER = 34962,
		// contains indices
		ELEMENT_ARRAY_BUFFER = 34963
	};

	byte* Data;
	int DataSize;
	int ComponentType;
	int Count;
	const char* Type;
	BufferTargetType TargetType;
	bool bNormalized;

	// Data for filling buffer
	byte* FillPtr;
#if MAX_DEBUG
	int FillCount;
	int ItemSize;
#endif

	FString BoundsMin;
	FString BoundsMax;

	BufferData()
	: Data(NULL)
	, DataSize(0)
	{}

	~BufferData()
	{
		if (Data) appFree(Data);
	}

	void Setup(int InCount, const char* InType, int InComponentType, int InItemSize, BufferTargetType InTargetType, bool InNormalized = false)
	{
		Count = InCount;
		Type = InType;
		TargetType = InTargetType;
		bNormalized = InNormalized;
		ComponentType = InComponentType;
		DataSize = InCount * InItemSize;
		// Align all buffers by 4, as requested by glTF format
		DataSize = Align(DataSize, 4);
		// Use aligned alloc for CVec4
		Data = (byte*) appMalloc(DataSize, 16);

		FillPtr = Data;
#if MAX_DEBUG
		FillCount = 0;
		ItemSize = InItemSize;
#endif
	}

	template<typename T>
	inline void Put(const T& p)
	{
#if MAX_DEBUG
		assert(sizeof(T) == ItemSize);
		assert(FillCount++ < Count);
#endif
		*(T*)FillPtr = p;
		FillPtr += sizeof(T);
	}

	bool IsSameAs(const BufferData& Other) const
	{
		// Compare metadata
		if (Count != Other.Count || strcmp(Type, Other.Type) != 0 || ComponentType != Other.ComponentType ||
			TargetType != Other.TargetType || bNormalized != Other.bNormalized || DataSize != Other.DataSize)
		{
			return false;
		}
		// Compare data
		return (memcmp(Data, Other.Data, DataSize) == 0);
	}
};

struct GLTFExportContext
{
	const char* MeshName;
	const CSkeletalMesh* SkelMesh;
	const CStaticMesh* StatMesh;

	TArray<BufferData> Data;

	GLTFExportContext()
	{
		memset(this, 0, sizeof(*this));
	}

	inline bool IsSkeletal() const
	{
		return SkelMesh != NULL;
	}

	// Compare last item of Data with other itest starting with FirstDataIndex, drop the data
	// if same data block found and return its index. If no matching data were found, return
	// index of that last data.
	int GetFinalIndexForLastBlock(int FirstDataIndex)
	{
		int LastIndex = Data.Num()-1;
		const BufferData& LastData = Data[LastIndex];
		for (int index = FirstDataIndex; index < LastIndex; index++)
		{
			if (LastData.IsSameAs(Data[index]))
			{
				// Found matching data
				Data.RemoveAt(LastIndex);
				return index;
			}
		}
		// Not found
		return LastIndex;
	}
};

#define VERT(n)		*OffsetPointer(Verts, (n) * VertexSize)

static void ExportSection(GLTFExportContext& Context, const CBaseMeshLod& Lod, int lodIdx, const CMeshVertex* Verts, int SectonIndex, FArchive& Ar)
{
	guard(ExportSection);

	int VertexSize = Context.IsSkeletal() ? sizeof(CSkelMeshVertex) : sizeof(CStaticMeshVertex);

	const CMeshSection& S = Lod.Sections[SectonIndex];
	bool bLast = (SectonIndex == Lod.Sections.Num()-1);

	int numMorphs = Context.IsSkeletal() ? Context.SkelMesh->Morphs.Num() : 0;

	// Remap section indices to local indices
	CIndexBuffer::IndexAccessor_t GetIndex = Lod.Indices.GetAccessor();
	TArray<int> indexRemap; // old vertex index -> new vertex index
	indexRemap.Init(-1, Lod.NumVerts);
	int numLocalVerts = 0;
	int numLocalIndices = S.NumFaces * 3;
	for (int idx = 0; idx < numLocalIndices; idx++)
	{
		int vertIndex = GetIndex(S.FirstIndex + idx);
		if (indexRemap[vertIndex] == -1)
		{
			indexRemap[vertIndex] = numLocalVerts++;
		}
	}

	// Prepare buffers
	int IndexBufIndex = Context.Data.AddZeroed();
	int PositionBufIndex = Context.Data.AddZeroed();
	int NormalBufIndex = Context.Data.AddZeroed();
	int TangentBufIndex = Context.Data.AddZeroed();
	TArray<int> MorphPositionBufIndex;
	TArray<int> MorphNormalBufIndex;
	if (numMorphs > 0)
	{
		for (int i = 0; i < numMorphs; i++)
		{
			MorphPositionBufIndex.Add(Context.Data.AddZeroed());
			MorphNormalBufIndex.Add(Context.Data.AddZeroed());
		}
	}

	int ColorBufIndex = -1;
	if (Lod.VertexColors)
	{
		ColorBufIndex = Context.Data.AddZeroed();
	}

	int BonesBufIndex[NUM_INFLUENCES / 4];
	int WeightsBufIndex[NUM_INFLUENCES / 4];
	if (Context.IsSkeletal())
	{
		for (int block = 0; block < NUM_INFLUENCES / 4; block++)
		{
			BonesBufIndex[block] = Context.Data.AddZeroed();
			WeightsBufIndex[block] = Context.Data.AddZeroed();
		}
	}
	else
	{
		for (int block = 0; block < NUM_INFLUENCES / 4; block++)
		{
			BonesBufIndex[block] = -1;
			WeightsBufIndex[block] = -1;
		}
	}

	int UVBufIndex[MAX_MESH_UV_SETS];
	for (int i = 0; i < Lod.NumTexCoords; i++)
	{
		UVBufIndex[i] = Context.Data.AddZeroed();
	}

	BufferData& IndexBuf = Context.Data[IndexBufIndex];
	BufferData& PositionBuf = Context.Data[PositionBufIndex];
	BufferData& NormalBuf = Context.Data[NormalBufIndex];
	BufferData& TangentBuf = Context.Data[TangentBufIndex];
	BufferData* UVBuf[MAX_MESH_UV_SETS];
	BufferData* ColorBuf = NULL;
	BufferData* BonesBuf[NUM_INFLUENCES / 4];
	BufferData* WeightsBuf[NUM_INFLUENCES / 4];
	TArray<BufferData*> MorphPositionBuf;
	TArray<BufferData*> MorphNormalBuf;

	PositionBuf.Setup(numLocalVerts, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::ARRAY_BUFFER);
	NormalBuf.Setup(numLocalVerts, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::ARRAY_BUFFER);
	TangentBuf.Setup(numLocalVerts, "VEC4", BufferData::FLOAT, sizeof(CVec4), BufferData::ARRAY_BUFFER);
	for (int i = 0; i < Lod.NumTexCoords; i++)
	{
		UVBuf[i] = &Context.Data[UVBufIndex[i]];
		UVBuf[i]->Setup(numLocalVerts, "VEC2", BufferData::FLOAT, sizeof(CMeshUVFloat), BufferData::ARRAY_BUFFER);
	}

	if (Lod.VertexColors)
	{
		ColorBuf = &Context.Data[ColorBufIndex];
		ColorBuf->Setup(numLocalVerts, "VEC4", BufferData::UNSIGNED_BYTE, 4, BufferData::ARRAY_BUFFER, /*InNormalized=*/ true);
	}

	if (Context.IsSkeletal())
	{
		for (int block = 0; block < NUM_INFLUENCES / 4; block++)
		{
			BonesBuf[block] = &Context.Data[BonesBufIndex[block]];
			WeightsBuf[block] = &Context.Data[WeightsBufIndex[block]];
			BonesBuf[block]->Setup(numLocalVerts, "VEC4", BufferData::UNSIGNED_SHORT, sizeof(uint16)*4, BufferData::ARRAY_BUFFER);
			WeightsBuf[block]->Setup(numLocalVerts, "VEC4", BufferData::UNSIGNED_BYTE, sizeof(uint32), BufferData::ARRAY_BUFFER, /*InNormalized=*/ true);
		}
	}
	else
	{
		for (int block = 0; block < NUM_INFLUENCES / 4; block++)
		{
			BonesBuf[block] = NULL;
			WeightsBuf[block] = NULL;
		}
	}
	for (int morph = 0; morph < numMorphs; morph++)
	{
		MorphPositionBuf.Add(&Context.Data[MorphPositionBufIndex[morph]]);
		MorphNormalBuf.Add(&Context.Data[MorphNormalBufIndex[morph]]);
	}

	// Prepare and build indices
	TArray<int> localIndices;
	localIndices.AddUninitialized(numLocalIndices);
	int* pIndex = localIndices.GetData();
	for (int i = 0; i < numLocalIndices; i++)
	{
		*pIndex++ = GetIndex(S.FirstIndex + i);
	}

	if (numLocalVerts <= 65536)
	{
		IndexBuf.Setup(numLocalIndices, "SCALAR", BufferData::UNSIGNED_SHORT, sizeof(uint16), BufferData::ELEMENT_ARRAY_BUFFER);
		for (int idx = 0; idx < numLocalIndices; idx++)
		{
			IndexBuf.Put<uint16>(indexRemap[localIndices[idx]]);
		}
	}
	else
	{
		IndexBuf.Setup(numLocalIndices, "SCALAR", BufferData::UNSIGNED_INT, sizeof(uint32), BufferData::ELEMENT_ARRAY_BUFFER);
		for (int idx = 0; idx < numLocalIndices; idx++)
		{
			IndexBuf.Put<uint32>(indexRemap[localIndices[idx]]);
		}
	}

	// Build reverse index map for fast lookup of vertex by its new index.
	// It maps new vertex index to old vertex index.
	TArray<int> revIndexMap;
	revIndexMap.AddUninitialized(numLocalVerts);
	for (int i = 0; i < indexRemap.Num(); i++)
	{
		int newIndex = indexRemap[i];
		if (newIndex != -1)
		{
			revIndexMap[newIndex] = i;
		}
	}

	// Build vertices
	for (int i = 0; i < numLocalVerts; i++)
	{
		int vertIndex = revIndexMap[i];
		const CMeshVertex& V = VERT(vertIndex);

		CVec3 Position = V.Position;

		CVec4 Normal, Tangent;
		Unpack(Normal, V.Normal);
		Unpack(Tangent, V.Tangent);
		// Unreal (and we are) using normal.w for computing binormal. glTF
		// uses tangent.w for that. Make this value exactly 1.0 of -1.0 to make glTF
		// validator happy.
	#if 0
		// There's some problem: V.Normal.W == 0x80 -> -1.008 instead of -1.0
		if (Normal.w > 1.001 || Normal.w < -1.001)
		{
			appError("%X -> %g\n", V.Normal.Data, Normal.w);
		}
	#endif
		Tangent.W = (Normal.W < 0) ? -1 : 1;

		TransformPosition(Position);
		TransformDirection(Normal);
		TransformDirection(Tangent);

		Normal.Normalize();
		Tangent.Normalize();

		// Fill buffers
		PositionBuf.Put(Position);
		NormalBuf.Put(Normal.XYZ);
		TangentBuf.Put(Tangent);
		UVBuf[0]->Put(V.UV);
	}

	// Compute bounds for PositionBuf
	CVec3 Mins, Maxs;
	ComputeBounds((CVec3*)PositionBuf.Data, numLocalVerts, sizeof(CVec3), Mins, Maxs);
	char buf[256];
	appSprintf(ARRAY_ARG(buf), "[ %1.9g, %1.9g, %1.9g ]", VECTOR_ARG(Mins));
	PositionBuf.BoundsMin = buf;
	appSprintf(ARRAY_ARG(buf), "[ %1.9g, %1.9g, %1.9g ]", VECTOR_ARG(Maxs));
	PositionBuf.BoundsMax = buf;

	if (Lod.VertexColors)
	{
		for (int i = 0; i < numLocalVerts; i++)
		{
			int vertIndex = revIndexMap[i];
			ColorBuf->Put(Lod.VertexColors[vertIndex]);
		}
	}

	if (Context.IsSkeletal())
	{
		for (int i = 0; i < numLocalVerts; i++)
		{
			int vertIndex = revIndexMap[i];
			const CMeshVertex& V0 = VERT(vertIndex);
			const CSkelMeshVertex& V = static_cast<const CSkelMeshVertex&>(V0);

			int16 Bones[NUM_INFLUENCES / 4][4];
			static_assert(sizeof(Bones) == sizeof(V.Bone), "Unexpected V.Bones size");
			memcpy(Bones, V.Bone, sizeof(Bones));
			for (int j = 0; j < NUM_INFLUENCES; j++)
			{
				// We have INDEX_NONE as list terminator, should replace with something else for glTF
				if (Bones[j / 4][j % 4] == INDEX_NONE)
				{
					Bones[j / 4][j % 4] = 0;
				}
			}

			for (int block = 0; block < NUM_INFLUENCES / 4; block++)
			{
				BonesBuf[block]->Put(*(uint64*)&Bones[block]);
				WeightsBuf[block]->Put(V.PackedWeights[block]);
			}
		}
	}

	// Secondary UVs
	for (int uvIndex = 1; uvIndex < Lod.NumTexCoords; uvIndex++)
	{
		BufferData* pBuf = UVBuf[uvIndex];
		const CMeshUVFloat* srcUV = Lod.ExtraUV[uvIndex-1];
		for (int i = 0; i < numLocalVerts; i++)
		{
			int vertIndex = revIndexMap[i];
			pBuf->Put(srcUV[vertIndex]);
		}
	}

	for (int morph = 0; morph < numMorphs; morph++)
	{
		// future optimization: output buffers should be sparse (GLTF Sec. 3.6.2.3)

		const CMorphLod& morphLod = Context.SkelMesh->Morphs[morph]->Lods[lodIdx];
		CVec3 zeroVec;
		zeroVec.Set(0.0f, 0.0f, 0.0f);
		TArray<CVec3> morphPositions;
		morphPositions.Init(zeroVec, numLocalVerts);
		TArray<CVec3> morphNormals;
		morphNormals.Init(zeroVec, numLocalVerts);

		for (const CMorphVertex& vert : morphLod.Vertices)
		{
			int localIndex = indexRemap[vert.VertexIndex];
			if (localIndex == -1)
			{
				continue;
			}
			morphPositions[localIndex] = vert.PositionDelta;
			morphNormals[localIndex] = vert.NormalDelta;
		}

		MorphPositionBuf[morph]->Setup(numLocalVerts, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::ARRAY_BUFFER);
		MorphNormalBuf[morph]->Setup(numLocalVerts, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::ARRAY_BUFFER);

		for (int i = 0; i < numLocalVerts; i++)
		{
			CVec3 PositionDelta = morphPositions[i];
			TransformPosition(PositionDelta);
			MorphPositionBuf[morph]->Put(PositionDelta);
			CVec3 NormalDelta = morphNormals[i];
			TransformDirection(NormalDelta);
			MorphNormalBuf[morph]->Put(NormalDelta);
		}

		ComputeBounds((CVec3*)MorphPositionBuf[morph]->Data, numLocalVerts, sizeof(CVec3), Mins, Maxs);
		appSprintf(ARRAY_ARG(buf), "[ %1.9g, %1.9g, %1.9g ]", VECTOR_ARG(Mins));
		MorphPositionBuf[morph]->BoundsMin = buf;
		appSprintf(ARRAY_ARG(buf), "[ %1.9g, %1.9g, %1.9g ]", VECTOR_ARG(Maxs));
		MorphPositionBuf[morph]->BoundsMax = buf;
	}

	// Write primitive information to json
	Ar.Printf(
		"        {\n"
		"          \"attributes\" : {\n"
		"            \"POSITION\" : %d,\n"
		"            \"NORMAL\" : %d,\n"
		"            \"TANGENT\" : %d,\n",
		PositionBufIndex, NormalBufIndex, TangentBufIndex
	);
	if (Lod.VertexColors)
	{
		Ar.Printf(
			"            \"COLOR_0\" : %d,\n",
			ColorBufIndex
		);
	}
	if (Context.IsSkeletal())
	{
		for (int block = 0; block < NUM_INFLUENCES / 4; block++)
		{
			Ar.Printf(
				"            \"JOINTS_%d\" : %d,\n"
				"            \"WEIGHTS_%d\" : %d,\n",
				block, BonesBufIndex[block],
				block, WeightsBufIndex[block]
			);
		}
	}
	for (int i = 0; i < Lod.NumTexCoords; i++)
	{
		Ar.Printf(
			"            \"TEXCOORD_%d\" : %d%s\n",
			i, UVBufIndex[i], i < (Lod.NumTexCoords-1) ? "," : ""
		);
	}
	Ar.Printf(
		"          },\n");
	if (Context.IsSkeletal() && Context.SkelMesh->Morphs.Num() > 0)
	{
		Ar.Printf(
			"          \"targets\" : [\n");
		for (int morph = 0; morph < numMorphs; morph++)
		{
			Ar.Printf(
				"            {\n"
				"              \"POSITION\" : %d,\n"
				"              \"NORMAL\" : %d\n"
				"            }%s\n",
				MorphPositionBufIndex[morph],
				MorphNormalBufIndex[morph],
				morph < numMorphs - 1 ? "," : "");
		}
		Ar.Printf(
			"          ],\n");
	}

	Ar.Printf(
		"          \"indices\" : %d,\n"
		"          \"material\" : %d\n"
		"        }%s\n",
		IndexBufIndex, SectonIndex,
		SectonIndex < (Lod.Sections.Num()-1) ? "," : ""
	);

	unguard;
}

struct CMat4
{
	float v[16];

	CMat4(const CCoords& C)
	{
		v[0] = C.axis[0][0];
		v[1] = C.axis[0][1];
		v[2] = C.axis[0][2];
		v[3] = 0;
		v[4] = C.axis[1][0];
		v[5] = C.axis[1][1];
		v[6] = C.axis[1][2];
		v[7] = 0;
		v[8] = C.axis[2][0];
		v[9] = C.axis[2][1];
		v[10] = C.axis[2][2];
		v[11] = 0;
		v[12] = C.origin[0];
		v[13] = C.origin[1];
		v[14] = C.origin[2];
		v[15] = 1;
	}
};

static void ExportSkinData(GLTFExportContext& Context, const CSkelMeshLod& Lod, FArchive& Ar)
{
	guard(ExportSkinData);

	int numBones = Context.SkelMesh->RefSkeleton.Num();

	int MatrixBufIndex = Context.Data.AddZeroed();
	BufferData& MatrixBuf = Context.Data[MatrixBufIndex];
	MatrixBuf.Setup(numBones, "MAT4", BufferData::FLOAT, sizeof(CMat4), BufferData::OTHER_BUFFER);

	Ar.Printf(
		"  \"nodes\" : [\n"
		"    {\n"
		"      \"name\" : \"%s\",\n"
		"      \"mesh\" : 0,\n"
		"      \"skin\" : 0,\n"
		"      \"children\" : [ 1 ]\n"
		"    },\n",
		Context.MeshName);

	TArray<CCoords> BoneCoords;
	BoneCoords.AddZeroed(numBones);

	for (int boneIndex = 0; boneIndex < numBones; boneIndex++)
	{
		const CSkelMeshBone& B = Context.SkelMesh->RefSkeleton[boneIndex];

		// Find all children
		TStaticArray<int, 32> children;
		for (int j = 0; j < numBones; j++)
		{
			if (boneIndex == j) continue;
			const CSkelMeshBone& B2 = Context.SkelMesh->RefSkeleton[j];
			if (B2.ParentIndex == boneIndex)
			{
				children.Add(j);
			}
		}

		Ar.Printf(
			"    {\n"
			"      \"name\" : \"%s\",\n",
			*B.Name
		);

		// Write children
		if (children.Num())
		{
			Ar.Printf("      \"children\" : [ %d", children[0]+FIRST_BONE_NODE);
			for (int j = 1; j < children.Num(); j++)
			{
				Ar.Printf(", %d", children[j]+FIRST_BONE_NODE);
			}
			Ar.Printf(" ],\n");
		}

		// Bone transform
		CVec3 bonePos = B.Position;
		CQuat boneRot = B.Orientation;
		if (boneIndex == 0)
		{
			boneRot.Conjugate();
		}

		TransformPosition(bonePos);
		TransformRotation(boneRot);

		Ar.Printf(
			"      \"translation\" : [ %1.9g, %1.9g, %1.9g ],\n"
			"      \"rotation\" : [ %1.9g, %1.9g, %1.9g, %1.9g ]\n",
			bonePos[0], bonePos[1], bonePos[2],
			boneRot.X, boneRot.Y, boneRot.Z, boneRot.W
		);

		boneRot.W *= -1;

		CCoords& BC = BoneCoords[boneIndex];
		BC.origin = bonePos;
		boneRot.ToAxis(BC.axis);
		if (boneIndex)
		{
			// World coordinate
			BoneCoords[B.ParentIndex].UnTransformCoords(BC, BC);
		}
		CCoords InvCoords;
		InvertCoords(BC, InvCoords);

		CMat4 BC4x4(InvCoords);
		MatrixBuf.Put(BC4x4);

		// Closing brace
		Ar.Printf(
			"    }%s\n",
			boneIndex == (numBones-1) ? "" : ","
		);
	}

	// Close "nodes" array
	Ar.Printf("  ],\n");

	// Make "skins"
	Ar.Printf(
		"  \"skins\" : [\n"
		"    {\n"
		"      \"inverseBindMatrices\" : %d,\n"
		"      \"skeleton\" : 1,\n"
		"      \"joints\" : [",
		MatrixBufIndex
	);
	for (int i = 0; i < numBones; i++)
	{
		if ((i & 31) == 0) Ar.Printf("\n        ");
		Ar.Printf("%d%s", i+FIRST_BONE_NODE, (i == numBones-1) ? "" : ",");
	}
	Ar.Printf(
		"\n"
		"      ]\n"
		"    }\n"
		"  ],\n"
	);

	unguard;
}

static void ExportAnimations(GLTFExportContext& Context, FArchive& Ar)
{
	guard(ExportAnimations);

	const CAnimSet* Anim = Context.SkelMesh->Anim;
	int NumBones = Context.SkelMesh->RefSkeleton.Num();

	// Build mesh to anim bone map

	TArray<int> BoneMap;
	BoneMap.Init(-1, NumBones);
	TArray<int> AnimBones;
	AnimBones.Empty(NumBones);

	for (int i = 0; i < NumBones; i++)
	{
		const CSkelMeshBone &B = Context.SkelMesh->RefSkeleton[i];
		for (int j = 0; j < Anim->TrackBoneNames.Num(); j++)
		{
			if (!stricmp(B.Name, Anim->TrackBoneNames[j]))
			{
				BoneMap[i] = j;			// lookup CAnimSet bone by mesh bone index
				AnimBones.Add(i);		// indicate that the bone has animation
				break;
			}
		}
	}

	// Don't export empty animations array.
	if (!Anim->Sequences.Num())
		return;

	Ar.Printf(
		"  \"animations\" : [\n"
	);

	int FirstDataIndex = Context.Data.Num();

	// Iterate over all animations
	for (int SeqIndex = 0; SeqIndex < Anim->Sequences.Num(); SeqIndex++)
	{
		const CAnimSequence &Seq = *Anim->Sequences[SeqIndex];

		Ar.Printf(
			"    {\n"
			"      \"name\" : \"%s\",\n",
			*Seq.Name
		);

		struct AnimSampler
		{
			enum ChannelType
			{
				TRANSLATION,
				ROTATION,
				SCALE
			};

			int BoneNodeIndex;
			ChannelType Type;
			const CAnimTrack* Track;
		};

#if SUPPORT_SCALE_KEYS
		constexpr int samplersPerBone = 3;
#else
		constexpr int samplersPerBone = 2;
#endif

		TArray<AnimSampler> Samplers;
		Samplers.Empty(AnimBones.Num() * samplersPerBone);

		//!! Optimization:
		//!! 1. there will be missing tracks (AnimRotationOnly etc) - drop such samplers
		//!! 2. store all time tracks in a single BufferView, all rotation tracks in another, and all position track in 3rd one - this
		//!!    will reduce amount of BufferViews in json text (combine them by data type)

		// Prepare channels array
		Ar.Printf("      \"channels\" : [\n");
		bool bHasOutput = false;
		for (int BoneIndex = 0; BoneIndex < AnimBones.Num(); BoneIndex++)
		{
			int MeshBoneIndex = AnimBones[BoneIndex];
			int AnimBoneIndex = BoneMap[MeshBoneIndex];

			const CAnimTrack* Track = Seq.Tracks[AnimBoneIndex];
			if (!Track->HasKeys())
			{
				//todo: may be store a reference pose position then?
				continue;
			}

			int TranslationSamplerIndex = Samplers.Num();
			AnimSampler* Sampler = new (Samplers) AnimSampler;
			Sampler->Type = AnimSampler::TRANSLATION;
			Sampler->BoneNodeIndex = MeshBoneIndex + FIRST_BONE_NODE;
			Sampler->Track = Track;

			int RotationSamplerIndex = Samplers.Num();
			Sampler = new (Samplers) AnimSampler;
			Sampler->Type = AnimSampler::ROTATION;
			Sampler->BoneNodeIndex = MeshBoneIndex + FIRST_BONE_NODE;
			Sampler->Track = Track;

#if SUPPORT_SCALE_KEYS
			int ScaleSamplerIndex = -1;
			if (Track->KeyScale.Num() > 0)
			{
				ScaleSamplerIndex = Samplers.Num();
				Sampler = new (Samplers) AnimSampler;
				Sampler->Type = AnimSampler::SCALE;
				Sampler->BoneNodeIndex = MeshBoneIndex + FIRST_BONE_NODE;
				Sampler->Track = Track;
			}
#endif

			if (bHasOutput)
			{
				Ar.Printf(",\n");
			}

			// Print glTF information. Not using usual formatting here to make output a little bit more compact.
			Ar.Printf(
				"        { \"sampler\" : %d, \"target\" : { \"node\" : %d, \"path\" : \"%s\" } },\n",
				TranslationSamplerIndex, MeshBoneIndex + FIRST_BONE_NODE, "translation"
			);
			Ar.Printf(
				"        { \"sampler\" : %d, \"target\" : { \"node\" : %d, \"path\" : \"%s\" } }",
				RotationSamplerIndex, MeshBoneIndex + FIRST_BONE_NODE, "rotation"
			);
#if SUPPORT_SCALE_KEYS
			if (ScaleSamplerIndex >= 0)
			{
				Ar.Printf(
					",\n"
					"        { \"sampler\" : %d, \"target\" : { \"node\" : %d, \"path\" : \"%s\" } }",
					ScaleSamplerIndex, MeshBoneIndex + FIRST_BONE_NODE, "scale"
				);
			}
#endif
			bHasOutput = true;
		}

		if (bHasOutput)
		{
			Ar.Printf("\n");
		}
		Ar.Printf("      ],\n");

		// Prepare samplers
		Ar.Printf("      \"samplers\" : [\n");
		for (int SamplerIndex = 0; SamplerIndex < Samplers.Num(); SamplerIndex++)
		{
			const AnimSampler& Sampler = Samplers[SamplerIndex];

			// Prepare time array
			const TArray<float>* TimeArray = (Sampler.Type != AnimSampler::ROTATION) ? &Sampler.Track->KeyPosTime : &Sampler.Track->KeyQuatTime;
			if (TimeArray->Num() == 0)
			{
				// For this situation, use track's time array
				TimeArray = &Sampler.Track->KeyTime;
			}
			int NumKeys;
			switch (Sampler.Type)
			{
			case AnimSampler::TRANSLATION:
				NumKeys = Sampler.Track->KeyPos.Num();
				break;
			case AnimSampler::ROTATION:
				NumKeys = Sampler.Track->KeyQuat.Num();
				break;
			case AnimSampler::SCALE:
				NumKeys = Sampler.Track->KeyScale.Num();
				break;
			}

			int TimeBufIndex = Context.Data.AddZeroed();
			BufferData& TimeBuf = Context.Data[TimeBufIndex];
			TimeBuf.Setup(NumKeys, "SCALAR", BufferData::FLOAT, sizeof(float), BufferData::OTHER_BUFFER);

			// Compute RateScale. Take care of null Rate - this might be valid if animation has just 1 frame (pose)
			float RateScale = (Seq.Rate > 0.001f) ? 1.0f / Seq.Rate : 1.0f;
			float LastFrameTime = 0;
			if (TimeArray->Num() == 0 || NumKeys == 1)
			{
				// Fill with equally spaced values
				for (int i = 0; i < NumKeys; i++)
				{
					TimeBuf.Put(i * RateScale);
				}
				LastFrameTime = NumKeys-1;
			}
			else
			{
				for (int i = 0; i < TimeArray->Num(); i++)
				{
					TimeBuf.Put((*TimeArray)[i] * RateScale);
				}
				LastFrameTime = (*TimeArray)[TimeArray->Num()-1];
			}
			// Prepare min/max values for time track, it's required by glTF standard
			TimeBuf.BoundsMin = "[ 0 ]";
			char buf[64];
			appSprintf(ARRAY_ARG(buf), "[ %1.9g ]", LastFrameTime * RateScale);
			TimeBuf.BoundsMax = buf;

			// Try to reuse TimeBuf from previous tracks
			TimeBufIndex = Context.GetFinalIndexForLastBlock(FirstDataIndex);

			// Prepare data
			int DataBufIndex = Context.Data.AddZeroed();
			BufferData& DataBuf = Context.Data[DataBufIndex];
			switch (Sampler.Type)
			{
			case AnimSampler::TRANSLATION:
				// Translation track
				DataBuf.Setup(NumKeys, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::OTHER_BUFFER);
				for (int i = 0; i < NumKeys; i++)
				{
					CVec3 Pos = Sampler.Track->KeyPos[i];
					TransformPosition(Pos);
					DataBuf.Put(Pos);
				}
				break;
			case AnimSampler::ROTATION:
				// Rotation track
				DataBuf.Setup(NumKeys, "VEC4", BufferData::FLOAT, sizeof(CQuat), BufferData::OTHER_BUFFER);
				for (int i = 0; i < NumKeys; i++)
				{
					CQuat Rot = Sampler.Track->KeyQuat[i];
					TransformRotation(Rot);
					if (Sampler.BoneNodeIndex - FIRST_BONE_NODE == 0)
					{
						Rot.Conjugate();
					}
					DataBuf.Put(Rot);
				}
				break;
#if SUPPORT_SCALE_KEYS
			case AnimSampler::SCALE:
				// Scale track
				DataBuf.Setup(NumKeys, "VEC3", BufferData::FLOAT, sizeof(CVec3), BufferData::OTHER_BUFFER);
				for (int i = 0; i < NumKeys; i++)
				{
					CVec3 Scale = Sampler.Track->KeyScale[i];
					DataBuf.Put(Scale);
				}
				break;
#endif
			}

			// Try to reuse data block as well
			DataBufIndex = Context.GetFinalIndexForLastBlock(FirstDataIndex);

			// Write glTF info
			Ar.Printf(
				"        { \"input\" : %d, \"output\" : %d }%s\n",
				TimeBufIndex, DataBufIndex, SamplerIndex == Samplers.Num()-1 ? "" : ","
			);
		}
		Ar.Printf("      ]\n");

		Ar.Printf("    }%s\n", SeqIndex == Anim->Sequences.Num()-1 ? "" : ",");
	}

	Ar.Printf("  ],\n");

	unguard;
}

struct MaterialIndices
{
	int DiffuseIndex;
	int NormalIndex;
	int EmissiveIndex;
	int OcclusionIndex;
	int MaterialIndex;
	int SpecularIndex;
	int SpecPowerIndex;

	MaterialIndices()
	: DiffuseIndex(-1)
	, NormalIndex(-1)
	, EmissiveIndex(-1)
	, OcclusionIndex(-1)
	, MaterialIndex(-1)
	, SpecularIndex(-1)
	, SpecPowerIndex(-1)
	{}
};

#define EXPORT_GLTF_MATERIALS			1

#define EXPORT_GLTF_MATERIAL_DIFFUSE	1
#define EXPORT_GLTF_MATERIAL_NORMAL		1
#define EXPORT_GLTF_MATERIAL_EMISSIVE	1
#define EXPORT_GLTF_MATERIAL_AO			1
#define EXPORT_GLTF_MATERIAL_ROUGHMET	1
#define EXPORT_GLTF_MATERIAL_SPECULAR	1


static void ExportMaterials(GLTFExportContext& Context, FArchive& Ar, const CBaseMeshLod& Lod)
{
	guard(ExportMaterials);
#if !EXPORT_GLTF_MATERIALS
	Ar.Printf("  \"materials\" : [\n");
	for (int i = 0; i < Lod.Sections.Num(); i++)
	{
		const UUnrealMaterial *Mat = Lod.Sections[i].Material;
		char dummyName[64];
		appSprintf(ARRAY_ARG(dummyName), "dummy_material_%d", i);
		CVec3 Color = GetMaterialDebugColor(i);
		Ar.Printf(
			"    {\n"
			"      \"name\" : \"%s\",\n"
			"      \"pbrMetallicRoughness\" : {\n"
			"        \"baseColorFactor\" : [ %g, %g, %g, 1.0 ],\n"
			"        \"metallicFactor\" : 0.1,\n"
			"        \"roughnessFactor\" : 0.5\n"
			"      }\n"
			"    }%s\n",
			Mat ? Mat->Name : dummyName,
			Color[0], Color[1], Color[2],
			i == Lod.Sections.Num() - 1 ? "" : ","
		);
	}
	Ar.Printf("  ],\n");
#else
	const UObject* OriginalMesh = Context.IsSkeletal() ? Context.SkelMesh->OriginalMesh : Context.StatMesh->OriginalMesh;
	// Collect texture info
	// TODO: Assumes all textures are in local directory; Viewers don't seem to support paths with ".."; export again to local?
	#define PROC(Arg) \
		if (Params.Arg) \
		{ \
			const char *filename = GetExportFileName(OriginalMesh, "%s.png", Params.Arg->Name); \
			int index = Images.AddUnique(filename); \
			info.Arg ## Index = index; \
		}
	TArray<MaterialIndices> Materials;
	TArray<FString> Images;

	guard(ExportMaterials::CreateFiles);
	for (int i = 0; i < Lod.Sections.Num(); i++) {
		Materials.AddDefaulted();
		MaterialIndices& info = Materials[Materials.Num()-1];
		if (!Lod.Sections[i].Material)
			continue;
		CMaterialParams Params;
		Lod.Sections[i].Material->GetParams(Params);

#if EXPORT_GLTF_MATERIAL_DIFFUSE
		PROC(Diffuse);
#endif
#if EXPORT_GLTF_MATERIAL_NORMAL
		PROC(Normal);
#endif
#if EXPORT_GLTF_MATERIAL_EMISSIVE
		PROC(Emissive);
#endif
#if EXPORT_GLTF_MATERIAL_AO
		PROC(Occlusion);
#endif
#if EXPORT_GLTF_MATERIAL_ROUGHMET
		PROC(Material);
#endif
#if EXPORT_GLTF_MATERIAL_SPECULAR
		PROC(Specular);
		PROC(SpecPower);
#endif
	}
	unguard;
	#undef PROC

	guard(ExportMaterials::WriteImages);
	if (Images.Num())
	{
		Ar.Printf("  \"images\" : [\n" );
		for (int i = 0; i < Images.Num(); i++)
		{
			const char *relativeName = strrchr(*Images[i], '/');
			if (!relativeName) relativeName = strrchr(*Images[i], '\\');
			if (relativeName) relativeName++;
			else relativeName = *Images[i];
			Ar.Printf(
				"    {\n"
				"      \"uri\" : \"%s\"\n"
				"    }%s\n",
				relativeName,
				i == Images.Num() - 1 ? "" : ","
			);
		}
		Ar.Printf("  ],\n");

		// texture index maps directly to image index
		Ar.Printf("  \"textures\" : [\n");
		for (int i = 0; i < Images.Num(); i++)
		{
			Ar.Printf(
				"    { \"source\": %d }%s\n",
				i, i == Images.Num() - 1 ? "" : ","
			);
		}
		Ar.Printf("  ],\n");
	}
	unguard;

	Ar.Printf("  \"materials\" : [\n");
	bool usedSpecularAndIOR = false;
	for (int i = 0; i < Lod.Sections.Num(); i++)
	{
		const UUnrealMaterial* Mat = Lod.Sections[i].Material;
		MaterialIndices& info = Materials[i];
		char dummyName[64];
		appSprintf(ARRAY_ARG(dummyName), "dummy_material_%d", i);

		Ar.Printf(
			"    {\n"
			"      \"name\" : \"%s\",\n"
			"      \"pbrMetallicRoughness\" : {\n",
			Mat ? Mat->Name : dummyName
		);

		if (info.MaterialIndex >= 0)
		{
			Ar.Printf(
				"        \"metallicRoughnessTexture\" : {\n"
				"          \"index\" : %d,\n"
				"          \"texCoord\" : %d\n"
				"        }",
				info.MaterialIndex, 0
			);
		}
		else if (info.SpecularIndex >= 0 || info.SpecPowerIndex >= 0)
		{
			Ar.Printf(
				"        \"metallicFactor\" : 0.0,\n"
				"        \"roughnessFactor\" : 1.0"
			);
		}
		else
		{
			Ar.Printf(
				"        \"metallicFactor\" : 1.0,\n"
				"        \"roughnessFactor\" : 1.0"
			);
		}

		if (info.DiffuseIndex >= 0)
		{
			Ar.Printf(
				",\n"
				"        \"baseColorTexture\" : {\n"
				"          \"index\" : %d,\n"
				"          \"texCoord\" : %d\n"
				"        }",
				info.DiffuseIndex, 0
			);
		}
		else
		{
			CVec3 Color = GetMaterialDebugColor(i);
			Ar.Printf(
				",\n"
				"        \"baseColorFactor\" : [ %1.9g, %1.9g, %1.9g, 1.0 ]\n",
				Color[0], Color[1], Color[2]
			);
		}

		Ar.Printf("\n      }");
		// end of pbrMetallicRoughness

		if (info.NormalIndex >= 0)
		{
			Ar.Printf(
				",\n"
				"      \"normalTexture\" : {\n"
				"        \"index\": %d,\n"
				"        \"texCoord\" : %d,\n"
				"        \"scale\" : %g\n"

				"      }",
				info.NormalIndex,
				0,
				1.f
			);
		}

		if (info.EmissiveIndex >= 0)
		{
			Ar.Printf(
				",\n"
				"      \"emissiveFactor\" : [ 1.0, 1.0, 1.0 ],\n"
				"      \"emissiveTexture\" : {\n"
				"        \"index\" : %d,\n"
				"        \"texCoord\" : %d\n"
				"      }",
				info.EmissiveIndex,
				0
			);
		}

		if (info.OcclusionIndex >= 0)
		{
			Ar.Printf(
				",\n"
				"      \"occlusionTexture\" : {\n"
				"        \"index\" : %d,\n"
				"        \"texCoord\" : %d\n"
				"      }",
				info.OcclusionIndex,
				0
			);
		}

		if (info.SpecularIndex >= 0 || info.SpecPowerIndex >= 0)
		{
			usedSpecularAndIOR = true;
			Ar.Printf(
				",\n"
            	"      \"extensions\": {\n"
                "        \"KHR_materials_specular\": {\n"
			);
			if (info.SpecularIndex >= 0)
			{
				Ar.Printf(
					"          \"specularColorTexture\": {\n"
					"            \"index\" : %d,\n"
					"            \"texCoord\" : %d\n"
					"          }",
					info.SpecularIndex,
					0
				);
			}
			else
			{
				// TODO: This use of SpecPower is based on UnRenderer.cpp. I'm not sure it is correct.
				// It probably also doesn't use the correct channel.
				Ar.Printf(
					"          \"specularTexture\": {\n"
					"            \"index\" : %d,\n"
					"            \"texCoord\" : %d\n"
					"          }",
					info.SpecPowerIndex,
					0
				);
			}
			Ar.Printf(
                "        },\n"
                "        \"KHR_materials_ior\": {\n"
                "          \"ior\": 0\n"
                "        }\n"
            	"      }"
			);
		}

		Ar.Printf("\n    }%s\n", i == Lod.Sections.Num() - 1 ? "" : ",");
	}
	Ar.Printf("  ],\n");

	if (usedSpecularAndIOR)
	{
		Ar.Printf(
			"  \"extensionsUsed\" : [\n"
			"    \"KHR_materials_specular\",\n"
			"    \"KHR_materials_ior\"\n"
			"  ],\n");
	}
#endif
	unguard;
}

static void ExportMeshLod(GLTFExportContext& Context, const CBaseMeshLod& Lod, int lodIdx, const CMeshVertex* Verts, FArchive& Ar, FArchive& Ar2)
{
	guard(ExportMeshLod);

	// Opening brace
	Ar.Printf("{\n");

	// Asset
	Ar.Printf(
		"  \"asset\" : {\n"
		"    \"generator\" : \"UE Viewer (umodel) build %s\",\n"
		"    \"version\" : \"2.0\"\n"
		"  },\n",
		STR(GIT_REVISION));

	// Scene
	Ar.Printf(
		"  \"scene\" : 0,\n"
		"  \"scenes\" : [\n"
		"    {\n"
		"      \"nodes\" : [ 0 ]\n"
		"    }\n"
		"  ],\n"
	);

	// Nodes
	if (!Context.IsSkeletal())
	{
		Ar.Printf(
			"  \"nodes\" : [\n"
			"    {\n"
			"      \"name\" : \"%s\",\n"
			"      \"mesh\" : 0\n"
			"    }\n"
			"  ],\n",
			Context.MeshName
		);
	}
	else
	{
		ExportSkinData(Context, static_cast<const CSkelMeshLod&>(Lod), Ar);
	}

	// Materials
	ExportMaterials(Context, Ar, Lod);

	// Meshes
	Ar.Printf(
		"  \"meshes\" : [\n"
		"    {\n"
		"      \"primitives\" : [\n"
	);
	for (int i = 0; i < Lod.Sections.Num(); i++)
	{
		ExportSection(Context, Lod, lodIdx, Verts, i, Ar);
	}
	Ar.Printf(
		"      ],\n"
		"      \"name\" : \"%s\"", // no newline yet, may need comma before extras
		Context.MeshName
	);
	int numMorphs = Context.IsSkeletal() ? Context.SkelMesh->Morphs.Num() : 0;
	if (numMorphs > 0)
	{
		Ar.Printf(",\n"
			"      \"extras\" : {\n"
			"        \"targetNames\" : [\n");
		for (int morph = 0; morph < numMorphs; morph++)
		{
			Ar.Printf(
				"          \"%s\"%s\n",
				*(Context.SkelMesh->Morphs[morph]->Name),
				morph < numMorphs - 1 ? "," : "");
		}
		Ar.Printf(
			"        ]\n"
			"      }\n");
	}
	else
	{
		Ar.Printf("\n");
	}
	Ar.Printf(
		"    }\n"
		"  ],\n"
	);

	// Write animations
	if (Context.IsSkeletal() && Context.SkelMesh->Anim)
	{
		ExportAnimations(Context, Ar);
	}

	// Write buffers
	int bufferLength = 0;
	for (int i = 0; i < Context.Data.Num(); i++)
	{
		bufferLength += Context.Data[i].DataSize;
	}

	Ar.Printf(
		"  \"buffers\" : [\n"
		"    {\n"
		"      \"uri\" : \"%s.bin\",\n"
		"      \"byteLength\" : %d\n"
		"    }\n"
		"  ],\n",
		Context.MeshName, bufferLength
	);

	// Write bufferViews
	Ar.Printf(
		"  \"bufferViews\" : [\n"
	);
	int bufferOffset = 0;
	for (int i = 0; i < Context.Data.Num(); i++)
	{
		const BufferData& B = Context.Data[i];
		Ar.Printf(
			"    {\n"
			"      \"buffer\" : 0,\n"
			"      \"byteOffset\" : %d,\n"
			"      \"byteLength\" : %d", // comma only if target follows, defer newline
			bufferOffset,
			B.DataSize
		);
		if (B.TargetType != BufferData::OTHER_BUFFER)
		{
			Ar.Printf(",\n"
				"      \"target\" : %d\n",
				(int)B.TargetType
			);
		}
		else
		{
			Ar.Printf("\n");
		}
		Ar.Printf(
			"    }%s\n",
			i == (Context.Data.Num()-1) ? "" : ","
		);
		bufferOffset += B.DataSize;
	}
	Ar.Printf(
		"  ],\n"
	);

	// Write accessors
	Ar.Printf(
		"  \"accessors\" : [\n"
	);
	for (int i = 0; i < Context.Data.Num(); i++)
	{
		const BufferData& B = Context.Data[i];
		Ar.Printf(
			"    {\n"
			"      \"bufferView\" : %d,\n",
			i
		);
		if (B.bNormalized)
		{
			Ar.Printf("      \"normalized\" : true,\n");
		}
		if (B.BoundsMin.Len())
		{
			Ar.Printf(
				"      \"min\" : %s,\n"
				"      \"max\" : %s,\n",
				*B.BoundsMin, *B.BoundsMax
			);
		}
		Ar.Printf(
			"      \"componentType\" : %d,\n"
			"      \"count\" : %d,\n"
			"      \"type\" : \"%s\"\n"
			"    }%s\n",
			B.ComponentType,
			B.Count,
			B.Type,
			i == (Context.Data.Num()-1) ? "" : ","
		);
	}
	Ar.Printf(
		"  ]\n"
	);

	// Write binary data
	guard(WriteBIN);
	for (int i = 0; i < Context.Data.Num(); i++)
	{
		const BufferData& B = Context.Data[i];
#if MAX_DEBUG
		assert(B.FillCount == B.Count);
#endif
		Ar2.Serialize(B.Data, B.DataSize);
	}
	unguard;

	// Closing brace
	Ar.Printf("}\n");

	unguard;
}

void ExportSkeletalMeshGLTF(const CSkeletalMesh* Mesh)
{
	guard(ExportSkeletalMeshGLTF);

	const UObject *OriginalMesh = Mesh->OriginalMesh;
	if (!Mesh->Lods.Num())
	{
		appNotify("Mesh %s has 0 lods", OriginalMesh->Name);
		return;
	}

	int MaxLod = (GExportLods) ? Mesh->Lods.Num() : 1;
	for (int Lod = 0; Lod < MaxLod; Lod++)
	{
		char suffix[32];
		suffix[0] = 0;
		if (Lod > 0)
		{
			appSprintf(ARRAY_ARG(suffix), "_Lod%d", Lod);
		}
		char meshName[256];
		appSprintf(ARRAY_ARG(meshName), "%s%s", OriginalMesh->Name, suffix);

		FArchive* Ar = CreateExportArchive(OriginalMesh, EFileArchiveOptions::TextFile, "%s.gltf", meshName);
		if (Ar)
		{
			GLTFExportContext Context;
			Context.MeshName = meshName;
			Context.SkelMesh = Mesh;

			FArchive* Ar2 = CreateExportArchive(OriginalMesh, EFileArchiveOptions::Default, "%s.bin", meshName);
			assert(Ar2);
			ExportMeshLod(Context, Mesh->Lods[Lod], Lod, Mesh->Lods[Lod].Verts, *Ar, *Ar2);
			delete Ar;
			delete Ar2;
		}
	}

	if (OriginalMesh->GetTypeinfo()->NumProps)
	{
		FArchive* PropAr = CreateExportArchive(OriginalMesh, EFileArchiveOptions::TextFile, "%s.props.txt", OriginalMesh->Name);
		if (PropAr)
		{
			OriginalMesh->GetTypeinfo()->SaveProps(OriginalMesh, *PropAr);
			delete PropAr;
		}
	}

	unguard;
}

void ExportStaticMeshGLTF(const CStaticMesh* Mesh)
{
	guard(ExportStaticMeshGLTF);

	UObject *OriginalMesh = Mesh->OriginalMesh;
	if (!Mesh->Lods.Num())
	{
		appNotify("Mesh %s has 0 lods", OriginalMesh->Name);
		return;
	}

	int MaxLod = (GExportLods) ? Mesh->Lods.Num() : 1;
	for (int Lod = 0; Lod < MaxLod; Lod++)
	for (int Lod = 0; Lod < MaxLod; Lod++)
	{
		char suffix[32];
		suffix[0] = 0;
		if (Lod > 0)
		{
			appSprintf(ARRAY_ARG(suffix), "_Lod%d", Lod);
		}
		char meshName[256];
		appSprintf(ARRAY_ARG(meshName), "%s%s", OriginalMesh->Name, suffix);

		FArchive* Ar = CreateExportArchive(OriginalMesh, EFileArchiveOptions::TextFile, "%s.gltf", meshName);
		if (Ar)
		{
			GLTFExportContext Context;
			Context.MeshName = meshName;
			Context.StatMesh = Mesh;

			FArchive* Ar2 = CreateExportArchive(OriginalMesh, EFileArchiveOptions::Default, "%s.bin", meshName);
			assert(Ar2);
			ExportMeshLod(Context, Mesh->Lods[Lod], Lod, Mesh->Lods[Lod].Verts, *Ar, *Ar2);
			delete Ar;
			delete Ar2;
		}
	}

	if (OriginalMesh->GetTypeinfo()->NumProps)
	{
		FArchive* PropAr = CreateExportArchive(OriginalMesh, EFileArchiveOptions::TextFile, "%s.props.txt", OriginalMesh->Name);
		if (PropAr)
		{
			OriginalMesh->GetTypeinfo()->SaveProps(OriginalMesh, *PropAr);
			delete PropAr;
		}
	}

	unguard;
}
