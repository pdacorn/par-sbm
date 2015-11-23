// This program is an implementation of the paper below.
// Chengbin Peng, Zhihua Zhang, Ka-Chun Wong, Xiangliang Zhang, David Keyes, "A scalable community detection algorithm for large graphs using stochastic block models", Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), Buenos Aires, Argentina, 2015.

#undef __FUNCT__
#define __FUNCT__ "main"
 


/* parameters for the algorithm */
#define STAGENUM 100 /* default 100, maximum number of stages. At the end of each stage, when stageId == 0, clusters are merged by the algorithm, and when stageId > 0, small clusters diluted into larger ones. */
#define ITERNUM 100 /* default 100, maximum number of iteration */
#define ALPHA_K 10 /* default 10, the coefficient to increase K to certain ammount. It is of no use if Kori is not specified. */



/* fixed parameters for the algorithm (change only when you understand) */
#define ITERNUMSMALL 100 /* default =ITERNUM; but if set smaller (less than 5, but should be >= 1 to update B after merge) may reduce some problem: avoid diag(B) becomes too small by too many damps; avoid more than one partitions in one community (especially in wiki data) */
#define MERGEFACTOR 1 /* default 1. It should be in [0,1], and the larger, the larger communities it will finally generate */
#define NCMIN 1 /* default 1 or 20, the smallest cluster size; isolated node won't be merged, so the minimum is two */
#define MINCOMMDEN 0 /* default 0 (no effect). Remove communities whose density is less than MINCOMMDEN times of the middle of the average p and q */
#define BMINRATIO 0 /* default 0 (no lower bound, for ideal case) or 0.1 (B should be no less than 0.1*avg, for practice, to avoid a cluster being too large and sparse) */
#define BMAXRATIO FLT_MAX /* default FLT_MAX (no upper bound) or 2 (B should be no larger than 2*avg */
#define BKINIT 1 /* default 1 (no effect) or 0.1, a smaller value can reduce the probability of irrelevant-node gathering*/
#define FREQ_NODEUPDATE 0.01 /* default 0.01, make sure that data is synced every 1% of nodes are updated */
#define DAMPB 0.8 // default 0.8. A larger one (e.g. 0.99) will have the nodes tend to join larger clusters
#define DAMPS 0.8

/* parameters for the implementation (change only when you understand) */
#define LINE_MAX_LEN 10000 /* maximum number of line length from text file */
#define PSENDRECV_BUFF 1000000 /* buffer size for communication  */
#define PSENDRECV_START 1 /* the extra size at the beginning of each data message. The unit is sizeof(int). */
#define PSENDRECV_TERM 3 /* the extra size needed to indicate the termination of all the data messages. The unit is sizeof(int). */
#define PSENDRECV_MINSIZE 100 /* the minimum message data size, increase to achieve a higher bandwidth, but less syncronization. It should be less than PSENDRECV_BUFF - PSENDRECV_START - PSENDRECV_TERM - msgdatasz, where msgdatasz is the size for one-piece of data in the sending (all the pieces constitute the data part of the message) */
#define MAXBUFF 100 /* buffer size for data transmission, in the unit of size(int) */
#define TIMERNUM 50 /* number of duplicative timers needed, plus one for summation */ /* the first 15 for timing, and the next 14 for non-timing testing, the last 1 for summation, */
#define BINNUM 4 /* 10 - 1000 */
#define MAXN_FNORDER N /* default N. 1e6, or N; maximum of N above which FnReadFromFileNet will write to file and FnConvertOrder will read from file.  if N<MAXN_FNORDER, allocate and free nodeidNewOrder in the main function; otherwise, allocate and free nodeidNewOrder inside each sub function (FnReadFromFileNet) */
#define CONVGSNUM 5 /* determine the size of convgFlagS */
#define COMNCR 0.5 /* defualt 0.5. Should be >0 && <1, if = 1, no recomputing, the rate of recomputing Nc, which means after COMNCR*N nodes are computed in total, Nc should be updated. */
#define EVPARANUM 5
#define FTEMPARRSIZE 20 /* should be larger than 11, and maybe more upon demand */
#define MAXCOMMPATH 20
#define DIGDISP 5 /* number of digits to display */
#define FREQ_SENDRECVRATIO 1 /* default 1, no change; 10, receive is launched 10 times more than send. It should be no less than 1. */



#include "mpi.h"
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <math.h>
#include <time.h>
#include "float.h"


/*
 User-defined routines
 */
struct PreLocSpace
{
    int *count, *countnz /* the non-zero entries of *count vector */, countnnz /* the number of non-zero entries of *count vector */, *unq, *iTemp ;/* the two variables (countnz and unq) actually have the same meaning */
    float /* *term1, *term2,*/ *obj, *fTemp;
    int *arrNlocalB;
} ;

struct CommEle
{
    MPI_Request *reqs;
    MPI_Request *reqr;
    int **sbuff;
    int **rbuff;
    int *sIdx; /* the actual length of current buffer */
    int *sbuffNow;/* current buffer id in use. When sending the data, datasize is indicated by the entry 0 of each buffer. The unit is sizeof(int). */
    int *sbuffAlt;
    int *rbuffNow;/* current buffer id in use */
    int *rbuffAlt;
    int *unsentNode; /* each entry (0) corresponding to the mpi rank that source has dispatched the message to; 1 for not yet dispatched */
    int countr; /* number of the communication-completed receivers */
    int *reqs_com;
    int *reqr_com;/* the communication-completed receivers */
    int ncomm /* number of senders (the same as receivers) */, size /* number of MPI ranks */;


    /* count in Irecv can be less than that in Isend: The MPI Standard 2.2 describes this in section 3.2.4. Basically the receive buffer (i.e., the count of the receiver) must be at least as big as the message being received (i.e., the count of the sender). Otherwise an overflow error occurs. If the count of the sender is less than the count of the receiver then the remaining elements of the receive buffer are unmodified. */
} ;


struct SortEleI
{
    int value;
    int index;
};


struct SortEleF
{
    float value;
    int index;
};


struct Mat
{
    float **arrv;
    int** arr;
    int* narr /* maximum number of non-zeros */, *nnz /* current number of non-zeros */;
    int size[2], Istart /* the starting index */, Iend /* one larger than the ending index */, Nlocal;
    int isBooleanMat /* if it is a Boolean matrix, the entries are either 1 or 0. So when MatAssemble, values for the same entry will be considered only once. If not Boolean matrix, MatAssemble, values for the same entry will be added */;
    MPI_Comm comm;

};


struct Queue
{
    int *arr;
    int size /* qend - qstart (if qend >= qstart) or qend - qstart + sizealloc (if qend < qstart) should be less than size. One block is wasting for convenience */, sizealloc /* the allocated space, one larger than size */;
    int qstart, qend /* qend is the exact ending location plus 1 */;
};




/* output: nzIdx, the nonzero index of Z, for the row corresponding to the Wrow*/
extern int UpdateZ2fastProf(int Wncols, const int *Wcols, float *Blog, float *Blog1mB, int *Nc, struct PreLocSpace *ppls, int *ZidxFul, int K, int N, int nodeID, int *nzIdx, int *compCounter, double *ptime, int rank);
extern int UpdateB(float *Barray, float dampB, int* Nc, int *Ni, int rank, int K, int N, int totalEdges, struct PreLocSpace *ppls, int IstartB, int IendB, int mode, float *qout);


extern int FnNodeRange(int  N, int  sizeW, int  rank, int *nodeStart, int *nodeEnd);
extern int FnComProcId(int nodeId, int * comProcId, int N, int size); /* determine the processor id for each node */
extern int PrintArrayI(const int size, const int *ar);
extern int PrintArrayR(const int size, const float *ar);
extern int FnPersistentCommInit(int size, struct CommEle* pcomme); /* the necessity of having a dedicated communication core on noor is illustrated here */
extern int FnPersistentCommDestroy(int size, struct CommEle* pcomme);
extern int FnReadFromFileConf(char* confPath, char* netPath, char * commPaths, int* commPathsN, char* outputPath, int *netMode, int *commMode, int *Kori, int *Kinit);
extern int FnReadFromFileNetN(int rank, int readProc, char* netPath, int* pN, int* pNmin, MPI_Comm *pcommAll, int isPrintWarning);
extern int FnReadFromFileNet(int rank, int readProc, int outputrank, char* netPath, int* pN, int* plocalN, struct Mat* pWlocal, int size, MPI_Comm *pcommAll, int* databuffs, int** databuffr_c, int* datasizer_c, struct CommEle* pcomme, char* outputPath, char* path, int isReorder, int *nodeidNewOrder);
extern int FnConvertOrder(int N, int **pZidxFul, char* outputPath, char* path, int isReorder, int rank, int readProc, int outputrank, const char *convertType, int *nodeidNewOrder);
extern int myGather(void *vec, int *dispgather, int *ngather, int rank,  MPI_Datatype datatype, MPI_Comm * comm);
extern int myGatherLocal(void *vecSendLocal, void *vec, int *dispgather, int *ngather, int rank, MPI_Datatype datatype, MPI_Comm * comm);
extern int myIGatherLocal(void *vecSendLocal, void *vec, int *dispgather, int *ngather, int rank, MPI_Datatype datatype, MPI_Comm * comm, MPI_Request *request);
extern int FnUniqueStart(struct PreLocSpace *pls, int nMax);
extern int FnUnique(int *vec, int ncols, int *ncolsNew, float *values, struct PreLocSpace *locT);
extern int FnUniqueTerm(struct PreLocSpace *pls);
extern int FnSendMessageStart(int *isEmpBuff, struct CommEle* pcomme);
extern int FnSendMessageBuff(int *pisEmpBuff, int* pchangeCounter, int* databuff, int datasize, int* dest, int destNum, int rank, struct CommEle* pcomme);
extern int FnSendMessageNow(int rank, struct CommEle* pcomme) ;
extern int	FnSendMessageNowProf(int rank, struct CommEle* pcomme, double *ptime);
extern int FnSendMessageTerm(int *isEmpBuff, int* termFlagLocal, int* termFlag, int* convgFlag, int tag, int changeCounter, int rank, int size, struct CommEle* pcomme);
extern int FnRecvMessageStart(int rank, int size, struct CommEle* pcomme);
extern int FnRecvMessageStartSome(int rank, int *sourceList , int sourceCount, struct CommEle* pcomme);
extern int FnRecvMessageProf(int* ptermFlag, int* pconvgFlag, int rank, int* pdatasizer, int** pdataloc_c, int* pdatasize_c, struct CommEle* pcomme, double *ptime);

int FnComputeComponent(struct Mat* adj, int i, int* plabelall, int labeli, struct Queue *q);

/* add T at the end of the function name to avoid confliction with Petsc */
int MatCreateT(struct Mat *pWlocal, int N1, int N2, int rowIdxStart, int rowIdxEnd, int *Wnnz, int isBooleanMat, MPI_Comm comm);
int MatSetValueT(struct Mat *pW, int idx1, int idx2, float value);
int  MatAssemble(struct Mat *pW);
int  MatDestroyT(struct Mat *pW);
int MatViewGlobal(struct Mat *pWlocal);




int FnQueueCreate(struct Queue *q, int size);
int FnQueuePush(struct Queue *q, int value);
int FnQueuePop(struct Queue *q, int* value, int* isSuccess);
int FnQueueView(struct Queue *q);
int FnQueueDestroy(struct Queue *q);


void random_number_init(unsigned int seed);
int random_number_int( int rMax);
float random_number_float();


int isFreqTrue(int* precvFreqIdx, int* precvFreq);
int FnTopNcIndices(int K, int Kori, int *Nc, int *orderIdx, int IstartB, int IendB, int NlocalB, float *Barray, float *Barray2, struct PreLocSpace *pls, int rank, MPI_Comm *pcommAll);


int max(int a, int b);
int min(int a, int b);
float minfloat(float a, float b);
int round_db(double number);
int sortint_index(int ncols, int* vec, int* orderIdx, int isChangeVec /* 1, change vec to be sorted; 0, don't change */); /* modified from PetscSortIntWithPermutation(ncols, vec, orderIdx); */
int FnCmpFunInt_index (const void * a, const void * b);
int sortfloat_index(int ncols, float* vec, int* orderIdx, int isChangeVec /* 1, change vec to be sorted; 0, don't change */); /* modified from PetscSortIntWithPermutation(ncols, vec, orderIdx); */
int FnCmpFunfloat_index (const void * a, const void * b);



int main(int argc,char **argv)
{


    int outputrank = -01/* -0 or -01, for convenience */, isPrintAdjMat = 0 /* set isPrintAdjMat = 1 to print the adjacency matrix */,  isDispResult = 0 /* 0 or 01. if 0, don't print the result (but will still write to file ) */;
    int isRandomUpdate = 1 /* default, 1; 1, the update order is randomized */, isRandomSeed = 1 /* default 1; 1, use a random seed generated by time(NULL) */, isReorderInput = 1 /* default 1; 1, reorder the matrix for load balance; 0, don't reorder */ ;
    float convgTol = 1e-3 /* default 1e-3, if the ratio of changed nodes below this Tolerance, converge */, convgTols = 1e-2 /* default 1e-3, the ratio below which the stage terminate due to too few cluster number changes */;
    int rank, i,j, i2, j2,  modid, iter,  dataidxTemp, flagT1 = 0, isRevert;
    int *dispgather, *dispgatherB, *ngather, *ngatherB, ngatherMin;
    int nodeId, N = 2, Nmin = 0, K, Kori = 0, Kinit = 0, Knew, Nlocal = 0, NlocalMin, NlocalB, comProcId, compCounter, maxDegreeLocal, *nodeidNewOrder;
    float  *buf, *Blog, *Blog1mB, dampB, dampS, dampConvg, isFinishStage, qout[2];
    int Istart, Iend, IstartB, IendB, *comProcT, nLowDense,  nLowDenseNode;
    struct Mat Wlocal, NdMat /* degrees caused by inter-cluster edges */, BmergeMat;
    char confPath[LINE_MAX_LEN] = "./conf.txt";
    char outputPath[LINE_MAX_LEN], path[LINE_MAX_LEN], pathTemp[LINE_MAX_LEN];   
    char netPath[LINE_MAX_LEN];
    char commPaths[MAXCOMMPATH*LINE_MAX_LEN];
    int commPathsN; 
    int size;
    int iternum, stageId, termFlag, termFlagLocal,  convgFlag, convgFlagLast, convgFlagBefore, termFlagB, termFlagLocalB,  convgFlagB, convgFlagS[CONVGSNUM*2] /* the last half space is for mpi_allreduce temporary use */;
    int flagexit;
    unsigned int randseed;
    int ncols, ncolsNew, *ncolsAr, *ncomAr, nzIdx = 0, sizeTMP, *rbuffi, *databuffs, *databuffr /* no need to allocate memory, use message buffer */, **databuffr_c, datasizes, datasizer, *datasizer_c, *iptrTemp;
    const int *cols;
    int **colsAr, **comAr, idxTemp, localIdx,  *bufferOcpArr, *dispTemp;
    const float *vals,   logEps = log(FLT_EPSILON);/* for MatGetRow */
    int *ZidxFul, *Nc /* total nodes in a community */, *Ni /* total degree caused by internal edges */, NiSum, *Nd /* total degree */, *NcLocalZ, *NiLocalZ, *NdLocalZ, *NcLocalZnew, *NiLocalZnew, *iTempPtr, *NcLocal, *NcLocalComm, *NiLocal, *NcNode, *NiNode, NcMin, NcMax, NcMinN, NcMaxN;
    float *Barray, *Barray2, *BarrayT;
    double timeLoading, timeLoadG, timeLoadGs = 0, timeInit, timeI, timeIs = 0, timeInitAll, timeInitG, timeInitGs = 0, timeRun, timeR2, timeRunAll, timeEvl,  timeT2, timeTemp, timeTempS[TIMERNUM];
    int  totalEdges, changeCounter, syncCounter, *updateOrderIdx, *mergeOrderIdx,  *orderIdx, iTemp, iTemp2, recvFreq /* determines how frequent to receive a message, in terms of the number of "while" iterations */, recvFreqIdx, recvFreqB, recvFreqBidx, sendFreqIdx, sendFreq, sendFreqBidx, sendFreqB, recompFreq; /* 32 bits in default */
    float *updateOrderRealar, fTemp, fTempArr[FTEMPARRSIZE], fTempArr2[FTEMPARRSIZE];
    FILE *fr, *fr1, *fr2;
    struct PreLocSpace pls, plsUpdateZ2 /* need a specific pls to improve speed in UpdateZ2fastProf */;
    struct CommEle comme;
    MPI_Comm commAll;
    int  readProc,  isEmpBuff;
    int netMode, commMode;
    int NcM, NcM2, ZM, ZM2;
    float pTemp, pTempMax, Lsame, Ldiff, lowDenseThreshold, avgComDensity;
    struct Queue que;


    dampB =  DAMPB; 
    dampS = DAMPS;
    dampConvg = 0.8;/* default, 0.8, to avoid ossilation of some nodes between two clusters; smaller value can help to terminate the osilation faster */


    MPI_Init(NULL, NULL);
    if (isRandomSeed == 1)
        random_number_init(-1);
    else
        random_number_init(0);

    timeLoading = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);





    readProc = 0; /* the process that reads file  */









    MPI_Comm_dup( MPI_COMM_WORLD, &commAll );


    flagexit = 0;
    if (rank == readProc)
    {
        Kori = 0;
        Kinit = 0;
        if (FnReadFromFileConf(confPath, netPath, commPaths, &commPathsN, outputPath, &netMode, &commMode, &Kori, &Kinit) < 0)
        {
            flagexit = 1;
        }
    }
    MPI_Bcast(&flagexit, 1, MPI_INT, readProc, commAll);
    if (flagexit == 1)
        return 0;









    FnPersistentCommInit(size, &comme);
    databuffs = malloc(sizeof(int)*MAXBUFF);
    /* databuffr = malloc(sizeof(int)*MAXBUFF);*/
    databuffr_c = malloc(sizeof(int*)*size*2);
    datasizer_c = malloc(sizeof(int)*size*2);




    if (FnReadFromFileNetN(rank, readProc, netPath, &N, &Nmin, &commAll, 1)<0)
        return 0;
    if (N <= MAXN_FNORDER /* && rank == readProc   */ && isReorderInput == 1)
    {
        if (outputrank >= 0)
            printf("[%d] nodeidNewOrder is allocated. \n", rank);
        nodeidNewOrder = malloc(sizeof(int)*N); 
    }



    if (FnReadFromFileNet(rank, readProc, outputrank, netPath, &N, &Nlocal, &Wlocal, size, &commAll, databuffs, databuffr_c, datasizer_c, &comme, outputPath, path, isReorderInput, nodeidNewOrder) < 0)
        return 0;

    if (rank == 0)
    {
        printf("The Adjacency Matrix is Loaded.\n");
    }


    if (rank == readProc)
    {





	
            if (Kori < 0 )
            {
                Kori = 0;
 			}
 

        /* # Kori = -1, determined by commPath file (if not exist, use Kori = 0). Kori = 0, determined by the algorithm. */


    }

    timeLoadG = MPI_Wtime();
    MPI_Bcast(&Kinit, 1, MPI_INT, readProc, commAll);
    MPI_Bcast(&Kori, 1, MPI_INT, readProc, commAll);
    MPI_Bcast(&commPathsN, 1, MPI_INT, readProc, commAll);
    timeLoadGs += MPI_Wtime() - timeLoadG;



    MPI_Barrier(commAll);
    timeLoading = MPI_Wtime() -  timeLoading;
    timeInitAll = MPI_Wtime();
    timeInit = MPI_Wtime();


    if (Kinit != 0) /* if there are inits, set inits */
        K = Kinit;
    else
    {
        if (Kori != 0) /* if no inits but has Kori, set by Kori*/
            K = Kori*ALPHA_K;
        else
            K = N;
    }
    
    if (K < 4)
        K = 4;/* for pls.ori allocation */

    Nc = malloc(sizeof(int)*K);/* number of non zero entries in each row of Ztrans */
    Ni = malloc(sizeof(int)*K);
    Nd = malloc(sizeof(int)*K);
    NcLocalZ = malloc(sizeof(int)*K);
    NiLocalZ = malloc(sizeof(int)*K);
    NdLocalZ = malloc(sizeof(int)*K);
    NcLocalZnew = malloc(sizeof(int)*K);
    NiLocalZnew = malloc(sizeof(int)*K);
    Barray = malloc(sizeof(float)*(K+1));
    Barray2 = malloc(sizeof(float)*(K+1));
    Blog = malloc(sizeof(float)*(K+1));
    Blog1mB = malloc(sizeof(float)*(K+1));



    /* setup displacement and local ranges, write into functions */
    ngather = malloc(sizeof(int)*size);
    dispgather = malloc(sizeof(int)*size);
    FnNodeRange(N, size, rank, &Istart, &Iend);
    Nlocal = Iend - Istart;
    ngather[rank] = Nlocal;
    timeI = MPI_Wtime();
    iTemp = ngather[rank];
    MPI_Allgather(&(iTemp), 1, MPI_INT, ngather, 1, MPI_INT, commAll);
    timeIs += MPI_Wtime() - timeI;
    ngatherMin = N;
    for (i = 0; i<size; i++)
    {
        if (ngatherMin > ngather[i] )
            ngatherMin = ngather[i];
    }
    NlocalMin = ngatherMin;

    /*  ngather[rank] = 0;*/
    dispgather[0] = 0;



    ngatherB = malloc(sizeof(int)*size);
    dispgatherB = malloc(sizeof(int)*size);
    FnNodeRange(K, size, rank, &IstartB, &IendB);
    NlocalB =IendB - IstartB;
    ngatherB[rank] = NlocalB;

    timeI = MPI_Wtime();
    iTemp = ngatherB[rank];
    MPI_Allgather(&(iTemp), 1, MPI_INT, ngatherB, 1, MPI_INT, commAll);
    timeIs += MPI_Wtime() - timeI;
    /*  ngatherB[rank] = 0;*/
    dispgatherB[0] = 0;
    for (i = 1; i<size; i++)
    {
        dispgather[i] = dispgather[i-1] + ngather[i-1];
        dispgatherB[i] = dispgatherB[i-1] + ngatherB[i-1];
    }
    if (outputrank >= 0)
        printf("[%d]. W row range: %d to %d; B row range: %d to %d.  \n", rank, Istart, Iend-1, IstartB, IendB-1);
		




    /* end setup */
















    NcLocal = malloc(sizeof(int)*NlocalB);
    NcLocalComm = malloc(sizeof(int)*NlocalB);
    NiLocal = malloc(sizeof(int)*NlocalB);
    ZidxFul = malloc(sizeof(int)*N);
    updateOrderRealar = malloc(sizeof(float)*Nlocal);      
    pls.count = malloc(sizeof(int)*K);
    pls.countnz = malloc(sizeof(int)*K);
    pls.unq = malloc(sizeof(int)*K);
    pls.iTemp = malloc(sizeof(int)*K);
    pls.obj = malloc(sizeof(float)*K); /* its size should be at least 4 for later use */
    pls.fTemp = malloc(sizeof(float)*K);
    pls.arrNlocalB = malloc(sizeof(float)*NlocalB); /* its size should be at least 4 for later use */
    plsUpdateZ2.count = malloc(sizeof(int)*K);
    plsUpdateZ2.countnz = malloc(sizeof(int)*K);
    plsUpdateZ2.unq = malloc(sizeof(int)*K);
    plsUpdateZ2.iTemp = malloc(sizeof(int)*K);
    plsUpdateZ2.obj = malloc(sizeof(float)*K); /* its size should be at least 4 for later use */
    plsUpdateZ2.fTemp = malloc(sizeof(float)*K);
    plsUpdateZ2.arrNlocalB = malloc(sizeof(float)*NlocalB); /* its size should be at least 4 for later use */
    NcNode = malloc(sizeof(int)*Nlocal);
    NiNode = malloc(sizeof(int)*Nlocal);
    ncomAr = malloc(sizeof(int)*Nlocal);
    comAr = malloc(sizeof(int *)*Nlocal);
    updateOrderIdx = malloc(sizeof(int)*Nlocal);
    mergeOrderIdx = malloc(sizeof(int)*Nlocal);
    orderIdx = malloc(sizeof(int)*K);
    bufferOcpArr = malloc(sizeof(int)*size);
    dispTemp = malloc(sizeof(int)*size);

    pls.countnnz = 0;
    for (i = 0; i<K; i++)
        pls.count[i] = 0;
    plsUpdateZ2.countnnz = 0;
    for (i = 0; i<K; i++)
        plsUpdateZ2.count[i] = 0;






    totalEdges = 0;
    maxDegreeLocal = 0;


    colsAr = Wlocal.arr;
    ncolsAr = Wlocal.narr;
    for (i = Istart; i < Iend; i++)
    {
  


        idxTemp = i - Istart;
        ncols = Wlocal.narr[idxTemp];
        totalEdges += ncols;
        if (ncols > maxDegreeLocal)
            maxDegreeLocal = ncols;

   

    }




    iTemp = 0;
    comProcT = malloc(sizeof(int)*maxDegreeLocal);
    for (i = Istart; i < Iend; i++)
    {
        ncols = Wlocal.narr[i - Istart];
        cols = Wlocal.arr[i - Istart];
        iTemp += ncols;

        for (j = 0; j<ncols; j++)
        {
            FnComProcId(cols[j], &comProcId, N, size);
            comProcT[j] = comProcId;/* including the rank of the processor itself */
        }
        FnUnique(comProcT, ncols, &ncolsNew, NULL, NULL);


        idxTemp = i - Istart;
        ncomAr[idxTemp] = ncolsNew;
        sizeTMP = sizeof(int )*ncolsNew;
        comAr[idxTemp] = malloc(sizeTMP);
        memcpy(comAr[idxTemp], comProcT, sizeTMP);



    }
    if (rank == 0)
        printf("nlocal = %d, elocal = %d. \n", Iend - Istart, iTemp);
    free(comProcT);

   
    timeI = MPI_Wtime();
    MPI_Allreduce(&totalEdges, &iTemp, 1, MPI_INT, MPI_SUM , commAll);
    totalEdges = iTemp;
    timeIs += MPI_Wtime() - timeI;
    totalEdges = totalEdges/2;/* each edge is counted twice in the adjacency matrix */


    if (rank == 0)
    {
        printf("No. of Communities = %d (initial) %d (final), No. of Nodes = %d, No. of Edges = %d. \n", Kinit, Kori, N, totalEdges);
        printf("No. of Processes = %d. \n", size);
        
    }







    /* only need to initialize B once, at the beginning */
    for (i = IstartB; i<IendB; i++)
    {
        Barray[i] = 1*0.5 + random_number_float()*0.5;
    }
    
    Barray[K] = totalEdges*2/((double)N*N)*BKINIT;
    if (Barray[K] > 0.5)
        Barray[K] = Barray[K]/2;



    
    if (rank == 0)
        printf("Matrix B is Created. \n");
    if (outputrank >= 0)
    {
        myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
    }
    if (rank == outputrank)
        PrintArrayR(K+1, Barray);/* test */






    for (i = Istart; i < Iend; i++)
    {
        ZidxFul[i] = min(floor(i/((float)N/K)), K-1);
        updateOrderRealar[i-Istart] = random_number_float();
    }
    sortfloat_index(Nlocal,updateOrderRealar,&(ZidxFul[Istart]), 0);


    timeI = MPI_Wtime();
    myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
    timeIs += MPI_Wtime() - timeI;



    if (outputrank >= 0)
    {
        myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
        myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);

        myGather(Nc, dispgatherB, ngatherB, rank, MPI_INT,  &commAll);
        myGather(Ni, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
    }
    if (rank == outputrank)
    {
        printf("Initialization.");

        printf("Print: Barray %d\n", iter);
        for (j = 0; j < K+1; j++)
            printf(" %f", Barray[j]);
        printf("\n");
        printf("Print: ZidxFul %d\n", iter);
        for (j = 0; j < N; j++)
            printf(" %d", ZidxFul[j]);
        printf("\n");
    }





    timeInit = MPI_Wtime() - timeInit;

    MPI_Barrier(commAll);
    timeInitAll = MPI_Wtime() - timeInitAll;
    timeRun = MPI_Wtime();
    timeRunAll = MPI_Wtime();






    isFinishStage = 0;
    for (stageId = 0; stageId < STAGENUM; stageId ++)
    {

		Barray[K] = Barray[K]*BKINIT; /* to make the initialization smaller*/
		if (Barray[K] > 0.5)
			Barray[K] = Barray[K]/2;
        
        convgFlagBefore = N;


        /* because B is computed from the generated clusters, no need to initialize B */

        /* update Nc, Ni at the very beginning*/




        /* initialize */
        for (i=0; i<K; i++)
        {
            NcLocalZ[i] = 0;
            NiLocalZ[i] = 0;
        }
        for (i = Istart; i < Iend; i++)
        {
            idxTemp = i - Istart;


            NcLocalZ[ZidxFul[i]] = NcLocalZ[ZidxFul[i]]+1;
            for (j = 0; j<ncolsAr[idxTemp]; j++)
                if (ZidxFul[i] == ZidxFul[colsAr[idxTemp][j]])/* find an edge in the same cluster */
                    NiLocalZ[ZidxFul[i]] = NiLocalZ[ZidxFul[i]]+1;
        }
   
 
        MPI_Allreduce(NcLocalZ, Nc, K, MPI_INT, MPI_SUM , commAll);
        MPI_Allreduce(NiLocalZ, Ni, K, MPI_INT, MPI_SUM , commAll);
        memcpy ( NcLocal, &(Nc[IstartB]), sizeof(int )*NlocalB);/* keep NcLocal, to allow the change of Nc in real time */
        memcpy ( NcLocalComm, NcLocal, sizeof(int )*NlocalB);
        memcpy ( NiLocal, &(Ni[IstartB]), sizeof(int )*NlocalB);
   



        convgFlag = N;

        if (rank == 0)
        {


            NcMin = Nc[0];
            NcMax = Nc[0];
            NcMinN = 0;
            NcMaxN = 0;
            for (j = 0; j<K; j++)
            {

                if (NcMin > Nc[j])
                {
                    NcMin = Nc[j];
                    NcMinN = 0;
                }
                if (NcMin == Nc[j])
                    NcMinN ++;

                if (NcMax < Nc[j])
                {
                    NcMax = Nc[j];
                    NcMaxN = 0;
                }
                if (NcMax == Nc[j])
                    NcMaxN ++;

            }
            printf("Before stage %d, the smallest/largest cluster size (number): %d (%d), %d (%d). Total cluster number: %d. \n", stageId, NcMin, NcMinN, NcMax, NcMaxN, K);
        }
        MPI_Barrier(commAll);
  

		if (stageId < 1 || isFinishStage >= 0.5)
			iternum = ITERNUM; 
		else
			iternum = ITERNUMSMALL;
			
        for (iter = 0; iter< iternum; iter++)
        {

   

            for (i = IstartB; i <= IendB; i++)
            {




                if (i == IendB) /* go to the end of B (larger than the max possible IendB), and then finish the loop */
                    i = K; /* finish at Iend-1, and then update the Kth entry before finish */



                if (Barray[i] < FLT_EPSILON)
                {
                    Blog[i] = logEps;
                    continue;
                }
                else if (Barray[i] > 1-FLT_EPSILON)
                {
                    Blog1mB[i] = logEps;
                    continue;
                }

                Blog[i] = log(Barray[i]);
                Blog1mB[i] = log(1-Barray[i]);



            }
      

            /* it is possible  to replace global communication， that here only sends message to relevant processes by the groups they have owned  */
            myGather(Blog, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
            myGather(Blog1mB, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
            /* NcLocal and Nc are updated before all the iteration and at the end of each iteration, so no need here. it will be used by ZidxFul */
            myGatherLocal(NcLocal, Nc, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
 











            if (iter==0 || isRandomUpdate == 1)
            {
                for (j = 0; j < Nlocal; j++)
                    updateOrderIdx[j] = j;
            }

            if (isRandomUpdate == 1)
            {
                for (i = 0; i<Nlocal; i++)
                    updateOrderRealar[i] = random_number_float();


                sortfloat_index(Nlocal,updateOrderRealar,updateOrderIdx, 0);
            }

       







            i = -1;
            termFlag = -size;
            termFlagLocal = -1;
            convgFlag = 0;
            changeCounter = 0;

            FnSendMessageStart(&isEmpBuff, &comme);
            FnRecvMessageStart(rank, size, &comme);
            
            
            sendFreq = FREQ_NODEUPDATE*N/size; 
            datasizes = 3;
            recvFreq = min(sendFreq, PSENDRECV_BUFF/datasizes)/FREQ_SENDRECVRATIO;
            recompFreq = (int)(COMNCR * N/(float)size); /*  frequency of recomputing Nc */ /* sendFreq *(N/(double)convgFlagLast); */
            if (recompFreq < 1)
                recompFreq = 1;



            recvFreqIdx = 0;
            sendFreqIdx = 0;

   
            while(1)
            {




    
                if (i+1<Nlocal)
                {


                    if (isEmpBuff == 1) /* data are sent successfully */
                    {



                        i++;
                        nodeId = Istart + updateOrderIdx[i];

                        timeR2 = 0;
                        localIdx = updateOrderIdx[i];
    
                        UpdateZ2fastProf(ncolsAr[localIdx], colsAr[localIdx], Blog, Blog1mB, Nc, &plsUpdateZ2, ZidxFul, K, N, nodeId, &nzIdx, &compCounter, &timeR2, rank);
                        syncCounter += compCounter; /* no need now */
              

    
                        if (ZidxFul[nodeId] != nzIdx)
                        {
            
                            databuffs[0] = nodeId;
                            databuffs[1] = nzIdx;
                            databuffs[2] = ZidxFul[nodeId];
                            datasizes = 3;

                            
                            FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, comAr[localIdx], ncomAr[localIdx], rank, &comme);
              
             
                            Nc[nzIdx] ++;
                            Nc[ZidxFul[nodeId]] --;



                            for (iTemp = 0; iTemp < 2; iTemp ++)
                            {
                                if (iTemp == 0)
                                    FnComProcId(nzIdx, &iTemp2, K, size);
                                if (iTemp == 1)
                                    FnComProcId(ZidxFul[nodeId], &iTemp2, K, size);

                                if (iTemp2 == rank)
                                {
                                    if (iTemp == 0)
                                        (NcLocalComm[nzIdx - IstartB]) ++;
                                    if (iTemp == 1)
                                        (NcLocalComm[ZidxFul[nodeId] - IstartB]) --;
                                }
                                else
                                {
                                    databuffs[0] = N + (iTemp+1);
                                    FnSendMessageBuff(&isEmpBuff, NULL, databuffs, datasizes, &iTemp2, 1, rank, &comme);
                                    if (isEmpBuff == 0)
                                    {
                                        printf("Warning: It needs a larger buffer in comme (larger PSENDRECV_BUFF) or more frequent sending (smaller FREQ_NODEUPDATE). Otherwise, it may be blocked at myGatherLocal.\n");
                                        // too small buffer will get receiver clogged at myGatherLocal(NcLocalComm, Nc, dispgatherB, ngatherB, rank, MPI_INT, &commAll); 
                                        // it is especially true when the load is imbalanced
									}
                                    isEmpBuff = 1;/* ignore the current buffer status because data buffer is changed */
                                }
                            }





                            ZidxFul[nodeId] = nzIdx; /* locally changed first */


                        }
                    }
                    else /* send the data again */
                    {

                        FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, comAr[localIdx], ncomAr[localIdx], rank, &comme); /* don't hold when the frequency is true */
          

                    }


                    if (isFreqTrue(&sendFreqIdx, &sendFreq))
                    {
                        FnSendMessageNowProf(rank, &comme, &timeT2) ;
                        


                    }

                }

                /* debug */
                if (i+1 == Nlocal + 1)
                {

                    
                    MPI_Barrier(commAll);
                    flagT1 = 1;
                }



                
                if (i+1 >= Nlocal && termFlagLocal != 0)
                {


                    FnSendMessageTerm(&isEmpBuff, &termFlagLocal, &termFlag, &convgFlag, iter, changeCounter, rank, size, &comme);
                    
                    sendFreq = 1;
                    recvFreq = 1;

                    

                }
                else
                {

                    if (i + 1 < 0.9*ngatherMin /* do nothing if the iteration is in the last 10% */ && (i+1)%(recompFreq)==0)
                    {



                        myGatherLocal(NcLocalComm, Nc, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
              

                    }
                }


                if (isFreqTrue(&recvFreqIdx, &recvFreq) == 1) /* if meet the frequency, go! */
                {



					if (FnRecvMessageProf(&termFlag, &convgFlag, rank, &datasizer, databuffr_c, datasizer_c, &comme, &timeT2) == 1) /* if (FnRecvMessag == 0), continue */
                    {


                        for (iTemp = 0; iTemp < datasizer; iTemp++)
                        {

                            dataidxTemp = 0;
                            iptrTemp = databuffr_c[iTemp];
                            while(dataidxTemp < datasizer_c[iTemp])
                            {
                                
                                if (iptrTemp[dataidxTemp] < N)
                                {
                                    ZidxFul[iptrTemp[dataidxTemp]] = iptrTemp[dataidxTemp+1];
                                    Nc[iptrTemp[dataidxTemp+1]] ++;
                                    Nc[iptrTemp[dataidxTemp+2]] --;
                                    dataidxTemp += 3;
                                }

                                if (iptrTemp[dataidxTemp] >= N)
                                {

                        
                                    iptrTemp[dataidxTemp] -= N;

                                    if (iptrTemp[dataidxTemp]==1)
                                        (NcLocalComm[iptrTemp[dataidxTemp+1] - IstartB]) ++;

                                    if (iptrTemp[dataidxTemp]==2)
                                        (NcLocalComm[iptrTemp[dataidxTemp+2] - IstartB]) --;

                                    dataidxTemp += 3;
                                }

                            }

                            

                            
                        }
                        


                        
                    }
                    

                    

                }




                if (termFlag == 0 ) /* before this, you can continue on updating ZidxFul, rather than waiting in the while loop */
                    break;





            }



            /* assume NcLocal, NiLocal is initialized */
            for (i=0; i<K; i++)
            {
                NcLocalZnew[i] = 0;
                NiLocalZnew[i] = 0;
            }
            for (i = Istart; i < Iend; i++)
            {
                idxTemp = i - Istart;

                NcLocalZnew[ZidxFul[i]] = NcLocalZnew[ZidxFul[i]]+1;
                for (j = 0; j<ncolsAr[idxTemp]; j++)
                    if (ZidxFul[i] == ZidxFul[colsAr[idxTemp][j]])/* find an edge in the same cluster */
                        NiLocalZnew[ZidxFul[i]] = NiLocalZnew[ZidxFul[i]]+1;
            }

            /* UpdateB, method 1 starts: use messages */


            i = -1;
            termFlagB = -size;
            termFlagLocalB = -1;
            convgFlagB = 0;
            changeCounter = 0;
            FnSendMessageStart(&isEmpBuff, &comme);
            FnRecvMessageStart(rank, size, &comme);
            sendFreqB = PSENDRECV_BUFF; /* can be as big as possible */
            datasizes = 3;
            recvFreqB = min(sendFreqB, PSENDRECV_BUFF/datasizes); 
            sendFreqBidx = 0;
            recvFreqBidx = 0;
            while(1)
            {

                if (i+1<K)
                {
                    if (isEmpBuff == 1) /* data are sent successfully */
                    {

                        i++;

                        if (NcLocalZnew[i] != NcLocalZ[i] || NiLocalZnew[i] != NiLocalZ[i])
                        {
                            

                            databuffs[1] = NcLocalZnew[i] - NcLocalZ[i];
                            databuffs[2] = NiLocalZnew[i] - NiLocalZ[i];
                            databuffs[0] = i;

                            FnComProcId(i, &comProcId, K, size);
                            datasizes = 3;

                            FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, &comProcId, 1, rank, &comme); /* to update the changeCounter */

                          

                            if (comProcId == rank)
                            {
                                NcLocal[i-IstartB] += databuffs[1];
                                NiLocal[i-IstartB] += databuffs[2];
                            }
                        }
                    }
                    else
                    {

                        FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, &comProcId, 1, rank, &comme); /* to update the changeCounter */
                       
                    }


                    if (isFreqTrue(&sendFreqBidx, &sendFreqB))
                    {
                        
                        FnSendMessageNowProf(rank, &comme, &timeT2) ;
                        
                    }


                }

                if (i+1 >= K && termFlagLocalB != 0)
                {
                    
                    FnSendMessageTerm(&isEmpBuff, &termFlagLocalB, &termFlagB, &convgFlagB, ITERNUM+iter, changeCounter, rank, size, &comme);
                    

                    sendFreqB = 1;
                    recvFreqB = 1;

                }



                if (isFreqTrue(&recvFreqBidx, &recvFreqB) == 1) /* if meet the frequency, go! */
                {

                    
                    if (FnRecvMessageProf(&termFlagB, &convgFlagB, rank, &datasizer, databuffr_c, datasizer_c, &comme, &timeT2) == 1)
                    {
                        for (iTemp = 0; iTemp < datasizer; iTemp++)
                        {
                            dataidxTemp = 0;
                            iptrTemp = databuffr_c[iTemp];
                            while(dataidxTemp < datasizer_c[iTemp])
                            {
                                NcLocal[iptrTemp[dataidxTemp]-IstartB] += iptrTemp[dataidxTemp+1];
                                NiLocal[iptrTemp[dataidxTemp]-IstartB] += iptrTemp[dataidxTemp+2];
                                dataidxTemp += 3;


                            }

                        }


                    }




                   
                }






                if (termFlagB == 0 ) /* before this, you can continue on updating ZidxFul, rather than waiting in the while loop */
                    break;







            }



            memcpy ( &(Nc[IstartB]), NcLocal, sizeof(int )*NlocalB);
            memcpy ( &(Ni[IstartB]), NiLocal, sizeof(int )*NlocalB);
            UpdateB(Barray, dampB, Nc, Ni, rank, K, N, totalEdges, &pls, IstartB, IendB, 1, NULL);
            fTemp = 0;
            for(j = IstartB; j<IendB; j++)
                fTemp += Barray[j];
            pls.obj[2] = fTemp;
            pls.obj[3] = NlocalB;
            MPI_Barrier(commAll);
            MPI_Allreduce(pls.obj, pls.fTemp, 4, MPI_FLOAT, MPI_SUM, commAll);
            for (j = 0; j<2; j++)    pls.obj[j] = pls.fTemp[j];
            UpdateB(Barray, dampB, Nc, Ni, rank, K, N, totalEdges, &pls, IstartB, IendB, 2, qout);
 
            
            fTemp = pls.obj[2]/(pls.obj[3]+FLT_MIN);
            avgComDensity = fTemp;
            if (BMINRATIO != 0 || BMAXRATIO != FLT_MAX)
                for(j = IstartB; j<IendB; j++)
                {
                    if ( Barray[j] < BMINRATIO*fTemp )
                    {
               
                        Barray[j] = BMINRATIO*fTemp;
                    }
                    
                    if ( Barray[j] > BMAXRATIO*fTemp )
                    {

                        Barray[j] = BMAXRATIO*fTemp;
                    }




                }



            iTempPtr = NcLocalZnew;
            NcLocalZnew = NcLocalZ;
            NcLocalZ = iTempPtr;
            iTempPtr = NiLocalZnew;
            NiLocalZnew = NiLocalZ;
            NiLocalZ = iTempPtr;


     
	
 





            if (outputrank >= 0)
            if (outputrank >= 0)
            {
                myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
                myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);

                myGather(Nc, dispgatherB, ngatherB, rank, MPI_INT,  &commAll);
                myGather(Ni, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
            }
            if (rank == outputrank)
            {
     
                printf("Print: Barray %d\n", iter);
                for (j = 0; j < K+1; j++)
                    printf(" %f", Barray[j]);
                printf("\n");
                printf("Print: ZidxFul %d\n", iter);
                for (j = 0; j < N; j++)
                    printf(" %d", ZidxFul[j]);
                printf("\n");
            }



            if (rank == outputrank)
            {

                printf("Print: [%d] Nc %d\n", rank, iter);
                for (j = 0; j < K; j++)
                    printf(" %d", Nc[j]);
                printf("\n");
                printf("Print: [%d] Ni %d\n", rank, iter);
                for (j = 0; j < K; j++)
                    printf(" %d", Ni[j]);
                printf("\n");
            }










            if (rank == 0)
            {
                if (iter == 0)
                    printf("Number of changed nodes: \n");
                printf("%d ", convgFlag);
                if ((iter+1) % 20 == 0)
                    printf("\n");

            }
            convgFlagLast = convgFlag;
            convgFlagBefore = dampConvg*convgFlagBefore + (1-dampConvg)*convgFlag;




            if(termFlag == 0 ) /* second digit is 1, finish the algorithm*/
            {
                if ((isFinishStage == 0 && convgFlag <= convgTol*N) || (isFinishStage >= 0.5 && convgFlag <= convgTol*N*0 /* for last iteration */) || convgFlagBefore < convgFlag /* to avoid some nodes jumping forward and backward */)
                {
                    MPI_Barrier(commAll);
                    if (rank==0)
                    {
                        if ((iter+1) % 20 != 0)
                            printf("\n");
                        printf( "The algorithm converged after %d iterations in stage %d. \n", iter+1, stageId  );
                    }
                    break;
                }
            }









        }

		
		if (!((isFinishStage == 0 && convgFlag <= convgTol*N) || (isFinishStage >= 0.5 && convgFlag <= convgTol*N*0 /* for last iteration */) || convgFlagBefore < convgFlag /* to avoid some nodes jumping forward and backward */))
        {
            MPI_Barrier(commAll);
            if (rank==0)
            {
                if ((iter+1) % 20 != 0)
                    printf("\n");
                printf( "The algorithm terminates after %d iterations [not converged] in stage %d. \n", iter+1, stageId   );
            }

        }


   




        if ( isFinishStage >= 0.5)
            isFinishStage = 1;





        if (stageId >= 0  )
        {



            myGatherLocal(NcLocal, Nc, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
            myGatherLocal(NiLocal, Ni, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
         


    

            for (i = 0; i < K; i++)
                NdLocalZ[i] = 0;

            for (i = Istart; i < Iend; i++)
            {
                iTemp = ZidxFul[i];
                idxTemp = i - Istart;
                NdLocalZ[iTemp] += ncolsAr[idxTemp];

            }
            MPI_Allreduce(NdLocalZ, Nd, K, MPI_INT, MPI_SUM , commAll);
            MatCreateT(&NdMat, K, K, IstartB, IendB, &(Nd[IstartB]), 0, commAll);/* the final NaMat (only consider off-diagonal parts of the matrix) is miss match to the Nd */
       



            i = 0;
            j = -1;
            termFlag = -size;
            termFlagLocal = -1;
            convgFlag = 0;
            changeCounter = 0;

            FnSendMessageStart(&isEmpBuff, &comme);
            FnRecvMessageStart(rank, size, &comme);
 
            sendFreq = N/size; /* default 0.01, make sure that data is synced every 1% of nodes are updated */
            datasizes = 2;
            recvFreq = min(sendFreq, PSENDRECV_BUFF/datasizes);



            recvFreqIdx = 0;
            sendFreqIdx = 0;

            /* based on the messages for UpdateZ2fastProf */
            while(1)
            {

                if (i+1<Nlocal)
                {


                    if (isEmpBuff == 1) /* data are sent successfully */
                    {
                        j++;
                        if (j >= ncolsAr[i] )
                        {
                            j = -1;
                            i++;
                            continue;
                        }
                        nodeId = Istart + i;


                        if (ZidxFul[nodeId] != ZidxFul[colsAr[i][j]]) /* only send those that are not in the same cluster */
                        {


                            databuffs[0] = ZidxFul[nodeId];
                            databuffs[1] = ZidxFul[colsAr[i][j]];
                            datasizes = 2;





                            FnComProcId(ZidxFul[nodeId], &comProcId, K, size);
                            FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes,  &comProcId, 1, rank, &comme);

                            if (comProcId == rank)
                            {
                                MatSetValueT(&NdMat, databuffs[0] ,  databuffs[1], 1);
                            }


                        }
                    }
                    else /* send the data again */
                    {

                        FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, &comProcId, 1, rank, &comme); /* don't hold when the frequency is true */

                    }


                    if (isFreqTrue(&sendFreqIdx, &sendFreq))
                    {
                        FnSendMessageNowProf(rank, &comme, &timeT2) ;


                    }

                }





                if (i+1 >= Nlocal && termFlagLocal != 0)
                {


                    FnSendMessageTerm(&isEmpBuff, &termFlagLocal, &termFlag, &convgFlag, iter, changeCounter, rank, size, &comme);

                    sendFreq = 1;
                    recvFreq = 1;

                  

                }





                if (isFreqTrue(&recvFreqIdx, &recvFreq) == 1) /* if meet the frequency, go! */
                {


                    if (FnRecvMessageProf(&termFlag, &convgFlag, rank, &datasizer, databuffr_c, datasizer_c, &comme, &timeT2) == 1) /* if (FnRecvMessag == 0), continue */
                    {


                        for (iTemp = 0; iTemp < datasizer; iTemp++)
                        {

                            dataidxTemp = 0;
                            iptrTemp = databuffr_c[iTemp];
                            while(dataidxTemp < datasizer_c[iTemp])
                            {
                          
                                MatSetValueT(&NdMat,iptrTemp[dataidxTemp], iptrTemp[dataidxTemp+1], 1);
                                dataidxTemp += 2;


                            }
                          

                        }
                       

                    
                    }
                   



                }




                if (termFlag == 0 ) 
                    break;





            }
		
		    MatAssemble(&NdMat);        
            MPI_Barrier(commAll);
       
          

            nLowDense = 0;
            nLowDenseNode = 0;
            for (i = 0; i<CONVGSNUM; i++)
                convgFlagS[i] = 0;
            for (i = IstartB; i < IendB; i++)
            {
                localIdx = i - IstartB;
                pTempMax = -1;
                if ((stageId >= 1 || isFinishStage >= 0.5 /* before the last stage, merge small clusters */) && Nc[i] < NCMIN )
                    pTempMax = -FLT_MAX;
                pls.count[i] = -1;


                if (isFinishStage < 0.5)
                {
                    lowDenseThreshold = avgComDensity *MINCOMMDEN;
                    if ((float)(Ni[i])/(Nc[i]+FLT_EPSILON)/(Nc[i]+FLT_EPSILON) < lowDenseThreshold )
                    {
                        pls.count[i] = -2;
                        nLowDense += 1;
                        nLowDenseNode = Nc[i];
                        
                        continue;
                        
                        
                    }
                }
          

                for (j = 0; j < NdMat.narr[localIdx]; j++)
                {

                  
                  
                  
                    idxTemp = NdMat.arr[localIdx][j];
                    if (idxTemp == i)
                        continue;

                    NcM = Nc[i] * Nc[idxTemp]; 
                    ZM = NdMat.arrv[localIdx][j];

                    NcM2 = Nc[i] + Nc[idxTemp];  
                    NcM2 = NcM2*NcM2;
                    ZM2 = ZM*2 + Ni[i] + Ni[idxTemp]; 
                    
                    
                              fTemp = MERGEFACTOR; // from 0 to 1, the smaller the fTemp, the smaller the final clusters
                              pTemp = ZM2/(float)NcM2;
                              if (pTemp < FLT_EPSILON)
                    				pTemp = FLT_EPSILON;
                    		  if (pTemp > 1-FLT_EPSILON)
                    				pTemp = 1-FLT_EPSILON;
                              Lsame = ZM * log(pTemp) + (NcM - ZM)*log(1-pTemp);
                              
                              pTemp = (1-fTemp)*ZM/(float)NcM + fTemp * Barray[K];
                              if (pTemp < FLT_EPSILON)
                    				pTemp = FLT_EPSILON;
                    		  if (pTemp > 1-FLT_EPSILON)
                    				pTemp = 1-FLT_EPSILON;
                              Ldiff = ZM * log(pTemp) + (NcM - ZM)*log(1-pTemp);
                    

                
                    

                    if (Lsame - Ldiff > pTempMax  && Nc[idxTemp] != 0)
                    {
                        pTempMax = Lsame - Ldiff;

                        if ((stageId >= 1 || isFinishStage >= 0.5 /* before the last stage, merge small clusters */) && Nc[i] < NCMIN   )
                        {
                            if (pls.count[i] == -1)
                                convgFlagS[0] ++;

                            pls.count[i] = idxTemp ;
                            pls.count[i] = -2;
                            
                         
                        }
                        else if ((pTempMax > 0) ) /* if not the final stage, only do the possible merge */
                        {
                            
                            if (pls.count[i] == -1)
                                convgFlagS[1] ++;
                                
 
                            pls.count[i] = idxTemp;

                        }
                    }
                }
 
                if (NdMat.narr[localIdx] == 0 && Nc[i] < NCMIN )
                {
                    if (Nc[i] == 0)
                    {
                        if (pls.count[i] == -1)
                            convgFlagS[2] ++;

                        pls.count[i] = 0;

                    }
                    else
                    {
 

                        convgFlagS[3] ++;


                    }
                }


            }
          
            if (rank == 0 && nLowDense > 0)
              printf("Number of low density (less than %f edge density) communities (nodes) removed: %d (%d)\n", lowDenseThreshold, nLowDense, nLowDenseNode);

            MatDestroyT(&NdMat);

            



            for ( i = 0; i<K; i++)
            {
                pls.countnz[i] = 0;
            }
            for ( i = IstartB; i<IendB; i++)
            {
                if (pls.count[i] != -1 && pls.count[i] != -2)
                {
                    pls.countnz[i] ++; /* source and destination are added by one */
                    pls.countnz[pls.count[i]] ++;
                }
            }
            myGather(pls.count, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
            MPI_Allreduce(pls.countnz, pls.iTemp, K, MPI_INT, MPI_SUM , commAll);
            for (j = 0; j<K; j++)      pls.countnz[j] = pls.iTemp[j];
     
            MatCreateT(&BmergeMat, K, K, 0, K, pls.countnz, 1, MPI_COMM_SELF);
            for ( i = 0; i<K; i++)
            {


                if (pls.count[i] != -1 && pls.count[i] != -2)
                {
                    MatSetValueT(&BmergeMat, i, pls.count[i], 1);
                    MatSetValueT(&BmergeMat, pls.count[i], i, 1);
                }
            }
            MatAssemble(&BmergeMat);/* everyone has the same variable */



            
            iTemp = 0;
            for (i = 0; i<K; i++)
            {
                if (pls.count[i] == -2)
                {
                    pls.iTemp[iTemp] = i;
                    iTemp ++;
                }

                pls.count[i] = i; /* set index */

            }
            FnQueueCreate(&que, 2*K);/* the max buff needed is no larger than 2*K */
            for (i = IstartB; i<IendB; i++) /* only compute components connected to local nodes */
            {
                
                if (pls.count[i] == i) /* has not been visited */
                    FnComputeComponent(&BmergeMat, i, pls.count, pls.count[i], &que); /* for node i in graph BmergeMat, set all the nodes in the same component to to labeli */

            }
            for (i = 0; i<iTemp; i++)
                pls.count[pls.iTemp[i]] = -2;

            
            FnQueueDestroy(&que);
            MatDestroyT(&BmergeMat);
            MPI_Allreduce(pls.count, pls.iTemp, K, MPI_INT, MPI_MIN , commAll);
            for (j = 0; j<K; j++)  pls.count[j] = pls.iTemp[j];
    
           
            for (i=0; i<K; i++)
                NcLocalZ[i] = 0;
            for (i = Istart; i < Iend; i++)
            {
                idxTemp = i - Istart;
                NcLocalZ[ZidxFul[i]] = NcLocalZ[ZidxFul[i]]+1;
            }
            MPI_Allreduce(NcLocalZ, Nc, K, MPI_INT, MPI_SUM , commAll);
            MPI_Allreduce(convgFlagS, &(convgFlagS[CONVGSNUM]), CONVGSNUM, MPI_INT, MPI_SUM , commAll);
            memcpy(convgFlagS, &(convgFlagS[CONVGSNUM]), sizeof(int)* CONVGSNUM);
      

            fTemp = 0;

           
            myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
            for ( i = 0; i<K; i++)
                pls.countnz[i] = -1;
            Knew = -1;
            for ( i = 0; i<K; i++)
            {
                if (pls.count[i] == -2)
                    continue;

                if (pls.countnz[pls.count[i]] == -1 && Nc[pls.count[i]] > 0 )
                {
                    Knew++;
                    Barray2[Knew] = Barray[i];
                    pls.countnz[pls.count[i]] = Knew;/* determine new cluster id */

                }
                else
                {
                    pls.countnz[i] = pls.countnz[pls.count[i]]; /* set to be the new cluster id */
                    if (Nc[pls.countnz[i]] < Nc[i])
                        Barray2[pls.countnz[i]] = Barray[i];
                   
                }


              
            }



      

          

            Knew ++;


            if (rank == 0)
            {
                printf("Community number change is %d; Remaining number is %d.\n", Knew - K , Knew);
                
            }


        

            isRevert = 0;
            if (isFinishStage != 1)
            {
                if (Kori == 0)
                {

                    if (Knew == K /* no change */ || (abs(Knew - K) <  convgTols * K)/* small change */)

                        isFinishStage = 0.6;/* let it run again for final converge */

                }
                else
                {
                    if (Knew == K /* no change */ || (abs(Knew - K) <  convgTols * K)/* small change */)
                    {

                        isFinishStage = 0.7; /* allow it to have the last iteration */
                        if (Knew > Kori)/* when too big get converge, rerun*/
                            isFinishStage = 0.5;

                    }

                    
                    if (Knew < Kori)/* when it's too small, revert*/
                    {
                        isRevert = 1; /* no change */
                        isFinishStage = 0.5;
                    }


                }
                
                
                
                
                                 
                 
                
			}

          
            if (isRevert == 0)
            {

                Barray2[Knew] = Barray[K];
                BarrayT = Barray2; 
                Barray2 = Barray;
                Barray = BarrayT;


                
                for (i = Istart; i < Iend; i++)
                {
                   
                    if (pls.countnz[ZidxFul[i]] != -1)
                        ZidxFul[i] = pls.countnz[ZidxFul[i]];
                    else
                        ZidxFul[i] = random_number_int(Knew - 1);





                }
                myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
      

                K = Knew;


                FnNodeRange(K, size, rank, &IstartB, &IendB);
                NlocalB =IendB - IstartB;
                ngatherB[rank] = NlocalB;
                iTemp = ngatherB[rank];
                MPI_Allgather(&(iTemp), 1, MPI_INT, ngatherB, 1, MPI_INT, commAll);
                dispgatherB[0] = 0;
                for (i = 1; i<size; i++)
                {
                    dispgatherB[i] = dispgatherB[i-1] + ngatherB[i-1];
                }
            }
            else if (rank == 0)
                printf("[Due to the Revert] Community number change is %d; Remaining number is %d.\n", 0, K);

        }
       


        if ((isFinishStage == 0) && (stageId == STAGENUM - 2)) /* shrink to Kori and run again to finish*/
        {
            if (Kori != 0)
                isFinishStage = 0.5;
            else
                isFinishStage = 0.6;
        }


       
        /* when it stops, if K is still larger than Kori, the following approach starts */
        if (isFinishStage == 0.5 )
        {
            /* shrink to Kori according to size */
            Knew = Kori;


            if (Kori == 0) /* no need. for the case when isFinishStage == 0.6 */
            {
                iTemp = 0;
                for (i = IstartB; i<IendB; i++)
                {
                    if (Nc[i] < NCMIN)
                        iTemp ++;

                }
                MPI_Allreduce(&iTemp, &iTemp2, 1, MPI_INT, MPI_SUM , commAll);
                Knew = K - iTemp2;

            }

            /* update B and ZidxFul, shrink K */
            FnTopNcIndices(K, Knew, Nc, orderIdx, IstartB, IendB, NlocalB, Barray, Barray2, &pls, rank, &commAll); /* orderIdx only needs size Kori here */
        

          

            myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
            for (j = 0; j < Knew; j++)
            {

                Barray2[j] = Barray[orderIdx[j]];/* serial version: K-1-j*/
            }
            Barray2[Knew] = Barray[K];
            BarrayT = Barray2; /* exchange Barray2 and Barray */
            Barray2 = Barray;
            Barray = BarrayT;
     

            /* reuse a an index for ZidxFul */
            for (j = 0; j < K; j++)
                Barray2[j] = -1;
            for (j = 0; j < Knew; j++)
                Barray2[orderIdx[j]] = j;/* serial version: K-1-j*/
            for (i = 0; i < N; i++)
            {
                if (i >= Istart && i < Iend)
                {
                    if (Barray2[ZidxFul[i]] != -1)
                        ZidxFul[i] = Barray2[ZidxFul[i]];
                    else
                        ZidxFul[i] = random_number_int( Knew-1);

                }
                else
                    ZidxFul[i] = -1;




            }
            myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
      
            if (rank == 0)
            {
                if (Kori != 0)
                    printf("[Due to the Presenting of Original K] ");
                else
                    printf("[Due to the Lower Bound (%d) of Cluster Size] ", NCMIN);

                printf("Community number change is %d; Remaining number is %d.\n", Knew - K, Knew);
            }

            K = Knew;


            FnNodeRange(K, size, rank, &IstartB, &IendB);
            NlocalB =IendB - IstartB;
            ngatherB[rank] = NlocalB;
            iTemp = ngatherB[rank];
            MPI_Allgather(&(iTemp), 1, MPI_INT, ngatherB, 1, MPI_INT, commAll);
            dispgatherB[0] = 0;
            for (i = 1; i<size; i++)
            {
                dispgatherB[i] = dispgatherB[i-1] + ngatherB[i-1];
            }
   


            /* may handle the B by a single processor */
        }



    


        if (outputrank >= 0)
        {
            myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll);
            myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);

            myGather(Nc, dispgatherB, ngatherB, rank, MPI_INT,  &commAll);
            myGather(Ni, dispgatherB, ngatherB, rank, MPI_INT, &commAll);
        }
        if (rank == outputrank)
        {
            printf("At the end of a stage:\n");

            printf("Print: Barray %d\n", iter);
            for (j = 0; j < K+1; j++)
                printf(" %f", Barray[j]);
            printf("\n");
            printf("Print: ZidxFul %d\n", iter);
            for (j = 0; j < N; j++)
                printf(" %d", ZidxFul[j]);
            printf("\n");
        }



   




        if (isFinishStage == 1)
            break;


    }



  
    MPI_Waitall(comme.ncomm, comme.reqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(comme.ncomm, comme.reqr, MPI_STATUSES_IGNORE);


	timeRun = MPI_Wtime() -  timeRun;
    MPI_Barrier( commAll );
    timeRunAll = MPI_Wtime() -  timeRunAll;


    if (rank==0)
    {
        printf( "[ALL]. Data loading time is %f s.\n", timeLoading );/* initialization */
        printf( "[ALL]. Initialization time is %f s and running time is %f s.\n", timeInitAll, timeRunAll  );
    }




    for (i=0; i<K; i++)
    {
        NcLocalZ[i] = 0;
        NiLocalZ[i] = 0;
    }
    for (i = Istart; i < Iend; i++)
    {
        idxTemp = i - Istart;

        NcLocalZ[ZidxFul[i]] = NcLocalZ[ZidxFul[i]]+1;
        for (j = 0; j<ncolsAr[idxTemp]; j++)
            if (ZidxFul[i] == ZidxFul[colsAr[idxTemp][j]])/* find an edge in the same cluster */
                NiLocalZ[ZidxFul[i]] = NiLocalZ[ZidxFul[i]]+1;
    }
    MPI_Allreduce(NcLocalZ, Nc, K, MPI_INT, MPI_SUM , commAll);
    MPI_Allreduce(NiLocalZ, Ni, K, MPI_INT, MPI_SUM , commAll);
  
    myGather(Barray, dispgatherB, ngatherB, rank, MPI_FLOAT, &commAll);
    myGather(ZidxFul, dispgather, ngather, rank, MPI_INT,  &commAll); 
    FnConvertOrder(N, &ZidxFul, outputPath, path, isReorderInput, rank, readProc, outputrank, "ntoo", nodeidNewOrder);
    MPI_Bcast(ZidxFul, N, MPI_INT, 0, commAll);

    if (rank == 0)
    {


        if (convgFlagS[3] > 0)
            printf("Number of tiny isolated cluster [size < %d, and not connecting to other clusters]: %d.\n", NCMIN, convgFlagS[3]);



        if (K < 1000 && isDispResult == 1)
            printf("Nc (number of nodes in each cluster): \n");
        NcMin = Nc[0];
        NcMax = Nc[0];
        NcMinN = 0;
        NcMaxN = 0;
        iTemp = -1;
        for (j = 0; j<K; j++)
        {
            if (K < 1000 && isDispResult == 1)
                printf(" %d", Nc[j]);
 

            if (NcMin > Nc[j])
            {
                NcMin = Nc[j];
                NcMinN = 0;
            }
            if (NcMin == Nc[j])
                NcMinN ++;

            if (NcMax < Nc[j])
            {
                NcMax = Nc[j];
                NcMaxN = 0;
                iTemp = j;
            }
            if (NcMax == Nc[j])
                NcMaxN ++;


        }
        printf("Finally, the smallest/largest cluster size (number): %d (%d), %d (%d). Total cluster number: %d. \n", NcMin, NcMinN, NcMax, NcMaxN, K);

 


        if (K < 1000 && isDispResult == 1)
            printf("Ni (number of degrees caused by edges in each cluster): \n");
        NiSum = 0;
        for (j = 0; j<K; j++)
        {
            if (K < 1000 && isDispResult == 1)
                printf(" %d", Ni[j]);
            NiSum += Ni[j];
        }
        printf("\n");
        printf("Total degrees in all the clusters are %d, which is %f%% of total degrees. \n", NiSum, NiSum/2/(float)totalEdges*100);


        if (K < 1000 && isDispResult == 1)
            printf("Vector B ([p, q], p is the internal connection probability for different clusters, q is a scalar): \n");
        strcpy( path, outputPath );
        strcat( path, "resultB.dat");
        fr = fopen(path, "w");
        for (j = 0; j<K+1; j++)
        {
            if (K < 1000 && isDispResult == 1)
                printf(" %f", Barray[j]);
            fprintf(fr, "%d %f\n", j, Barray[j]);
        }
        fclose(fr);
        printf("\n");
 


        if (N<10000 && isDispResult == 1)
            printf("Vector communityID (ZidxFul, each entry represent the community ID starting from zero): \n");
        strcpy( path, outputPath );
        strcat( path, "resultZ.dat");
        fr = fopen(path, "w");
        for (j = 0; j<N; j++)
        {
            if (N<10000 && isDispResult == 1)
                printf(" %d", ZidxFul[j]);
            switch (commMode)
            {
				case 1:
					fprintf(fr, "%d\n", ZidxFul[j]);
					break;
				case 2:
					fprintf(fr, "%d %d\n", j+Nmin, ZidxFul[j]);
					break;
				default:
					printf("Undefined value: commMode = %d, which should be 1 or 2. ", commMode);
					
			}
        }
        fclose(fr);
        printf("\n");




    }

 


  

    MPI_Barrier(commAll);
    if (rank == 0)
        printf("All the computing work finishes. \n");
        
        

	

    free(databuffs);
    /* free(databuffr);*/
    free(databuffr_c);
    free(datasizer_c);

    FnPersistentCommDestroy(size, &comme);





    free(Nc);
    free(Ni);
    free(Nd);
    free(NcLocalZ);
    free(NiLocalZ);
    free(NdLocalZ);
    free(NcLocalZnew);
    free(NiLocalZnew);
    free(Blog);
    free(Blog1mB);
    free(Barray);
    free(Barray2);


    if (N <= MAXN_FNORDER  /* && rank == readProc   */ && isReorderInput == 1)
    {
        if (outputrank >= 0)
            printf("[%d] nodeidNewOrder is freed. \n", rank);
        free(nodeidNewOrder);
    }


    MatDestroyT(&Wlocal);
    for (i = Istart; i < Iend; i++)
    {
        idxTemp = i - Istart;

        
        free(comAr[idxTemp]);
    }
    free(updateOrderRealar);
    free(ZidxFul);
    free(pls.count);
    free(pls.countnz);
    free(pls.unq);
    free(pls.iTemp);

   
    free(pls.obj);
    free(pls.fTemp);
    free(pls.arrNlocalB);
    free(plsUpdateZ2.count);
    free(plsUpdateZ2.countnz);
    free(plsUpdateZ2.unq);
    free(plsUpdateZ2.iTemp);
    free(plsUpdateZ2.obj);
    free(plsUpdateZ2.fTemp);
    free(plsUpdateZ2.arrNlocalB);
    free(NcLocal);
    free(NcLocalComm);
    free(NiLocal);
    free(NcNode);
    free(NiNode);
    /*	free(ncolsAr);
     free(colsAr);*/
    free(ncomAr);
    free(comAr);
    free(updateOrderIdx);
    free(mergeOrderIdx);
    free(orderIdx);



    free(bufferOcpArr);
    free(dispTemp);












    free(ngather);
    free(dispgather);
    free(ngatherB);
    free(dispgatherB);

    MPI_Comm_free(&commAll);

    MPI_Finalize();
    return 0;
}






extern int UpdateZ2fastProf(int Wncols, const int *Wcols, float *Blog, float *Blog1mB, int *Nc, struct PreLocSpace *ppls, int *ZidxFul, int K, int N, int nodeID, int *nzIdx, int *compCounter, double *ptime, int rank)
{

    int i, iTemp, *count = (*ppls).count, *countnz = (*ppls).countnz, *pcountnnz = &((*ppls).countnnz), *unq = (*ppls).unq, unqNum, Zidxi = ZidxFul[nodeID], maxobjid;
    float *obj = (*ppls).obj, maxobj;


    *ptime = MPI_Wtime();
  

    /* implementation 2: time complexity O(countnnz) */
    for (i = 0; i<*pcountnnz; i++)
    {
        count[countnz[i]] = 0;
    }
    (*pcountnnz) = 0;



    for (i = 0; i<Wncols; i++)
    {
        iTemp = ZidxFul[Wcols[i]];
        if (count[iTemp] == 0)
        {
            countnz[*pcountnnz] = iTemp;
            (*pcountnnz) ++;
        }
        count[iTemp] += 1;
    }

    unqNum = 0;
    for (i = 0; i<*pcountnnz; i++)
    {
        unq[unqNum] = countnz[i];
        unqNum ++;
    }

    (*ptime) = MPI_Wtime() - (*ptime);
    /* (*ptime) = Wncols; /* debug */

    if (unqNum == 0)
    {
      
        return 0;
    }

    *compCounter = 1; /* at least 1, because ZidxFul may change even when unqNum == 1 */
    if (unqNum == 1)
    {
        *nzIdx = unq[0];
        return 0;
    }
    else
    {


        Nc[Zidxi] --;
        for (i = 0; i<unqNum; i++)
            Nc[unq[i]] -= count[unq[i]];
            

        maxobj = 0;
        for (i = 0; i<unqNum; i++)
        {

            obj[i] = count[unq[i]]*(Blog[unq[i]] - Blog[K]) + Nc[unq[i]]*(Blog1mB[unq[i]] - Blog1mB[K]);
       
            if (maxobj < obj[i] || i == 0)
            {
                maxobj = obj[i];
                maxobjid = i;
            }



        }


        Nc[Zidxi] ++;
        for (i = 0; i<unqNum; i++)
            Nc[unq[i]] += count[unq[i]];


        *nzIdx = unq[maxobjid]; /* IX in matlab*/



    }



    *compCounter = unqNum;
    return 0;
}



extern int	FnNodeRange(int  N, int  sizeW, int  rank, int *nodeStart, int *nodeEnd)
{

    *nodeEnd = round_db(N*((rank+1)/(double)sizeW))-1;
    *nodeStart = round_db(N*(rank/(double)sizeW));

    *nodeEnd = *nodeEnd + 1; 
    return 0;
}


int round_db(double number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}


extern int	 FnComProcId(int nodeId, int * comProcId, int N, int size) /* determine the processor id for each node */
{
   
    *comProcId =  ceil((nodeId+1)/(N/(double)size)) -1; 
    
    if (nodeId+1 <= round_db(N*(((*comProcId-1)+1)/(double)size))) /* by the plot, nodes may belong to the previous comProcId */
        *comProcId = *comProcId - 1;

    return 0;

}



extern int FnPersistentCommInit(int size, struct CommEle* pcomme)
{

    MPI_Request *reqs, *reqr;
    int **sbuff, **rbuff;
    float timeT, timeTs, timeT2, timeTs2, maxRecvRank, minRecvRank;
    int i, j, k, rank, iTemp;







    reqs = malloc(sizeof(MPI_Request)*size*2);
    reqr = malloc(sizeof(MPI_Request)*size*2);
    sbuff = malloc(sizeof(int*)*size*2);
    rbuff = malloc(sizeof(int*)*size*2);
    (*pcomme).sIdx = malloc(sizeof(int*)*size);
    (*pcomme).sbuffNow = malloc(sizeof(int*)*size);
    (*pcomme).sbuffAlt = malloc(sizeof(int*)*size);
    (*pcomme).rbuffNow = malloc(sizeof(int*)*size);
    (*pcomme).rbuffAlt = malloc(sizeof(int*)*size);
    (*pcomme).unsentNode = malloc(sizeof(int*)*size);
    (*pcomme).reqs_com = malloc(sizeof(int*)*size*2);
    (*pcomme).reqr_com = malloc(sizeof(int*)*size*2);
    for (i = 0; i<size; i++)
    {
        for (j = 0; j<2; j++)
        {
            iTemp = i+j*size;
            sbuff[iTemp] = malloc(sizeof(int)*PSENDRECV_BUFF);
            rbuff[iTemp] = malloc(sizeof(int)*PSENDRECV_BUFF);
           

        }
    }



    /* initialization */
    (*pcomme).reqs = reqs;
    (*pcomme).reqr = reqr;
    (*pcomme).sbuff = sbuff;
    (*pcomme).rbuff = rbuff;
    for ( i = 0; i<size; i++)
    {
        (*pcomme).sIdx[i] = PSENDRECV_START;
        for (j = 0; j<2; j++)
        {
            iTemp = i+j*size;
            sbuff[iTemp][0] = PSENDRECV_START;
            rbuff[iTemp][0] = PSENDRECV_START;
        }


        (*pcomme).sbuffNow[i] = i;
        (*pcomme).sbuffAlt[i] = i+size;
        (*pcomme).rbuffNow[i] = i;
        (*pcomme).rbuffAlt[i] = i+size;
        (*pcomme).unsentNode[i] = 0;


    }

    (*pcomme).ncomm = 2*size;
    (*pcomme).size = size;




    /* to initialize MPI_request, for MPI_Waitall in FnSendMessageStart */
    for ( i = 0; i<size; i++)
    {

        for (j = 0; j<2; j++)
        {
            iTemp = i+j*size;
            MPI_Isend(sbuff[iTemp], 0, MPI_INT, i, 0, MPI_COMM_WORLD, &(reqs[iTemp]));
            MPI_Irecv(rbuff[iTemp], 0, MPI_INT, i, 0, MPI_COMM_WORLD, &(reqr[iTemp]));
        }
    }


 

    return 0;
}

extern int FnPersistentCommDestroy(int size, struct CommEle* pcomme)
{
    int i, j, iTemp;

    for (i=0; i<size; i++)
    {
        for (j = 0; j<2; j++)
        {

            if (j == 0)
                iTemp = i;
            if (j == 1)
                iTemp = i+size;



            free((*pcomme).sbuff[iTemp]);
            free((*pcomme).rbuff[iTemp]);
           
        }
    }

    free((*pcomme).sbuff);
    free((*pcomme).rbuff);
    free((*pcomme).reqs);
    free((*pcomme).reqr);

    free((*pcomme).sIdx);
    free((*pcomme).sbuffNow);
    free((*pcomme).sbuffAlt);
    free((*pcomme).rbuffNow);
    free((*pcomme).rbuffAlt);
    free((*pcomme).unsentNode);
    free((*pcomme).reqs_com);
    free((*pcomme).reqr_com);

    return 0;
}


extern int FnReadFromFileConf(char* confPath, char* netPath, char* commPaths, int* commPathsN, char* outputPath, int *pnetMode, int *pcommMode, int *pKori, int *pKinit)
{

    int  Ktemp, Kmax, Kmin, lineNum, idxi, j;
    char line[LINE_MAX_LEN], varname[LINE_MAX_LEN], varval[LINE_MAX_LEN], line2[LINE_MAX_LEN];
    FILE *fr;
    int r, flag;




    fr = fopen (confPath, "rt");
    if (fr == NULL)
    {
        printf("Failed to open file: %s\n", confPath );
        return -1;
    }

    /* initialize paths */
    outputPath[0] = '\0';  
    netPath[0] = '\0';
    commPaths[0] = '\0';
    *commPathsN = 0;

	outputPath[0] = '.'; outputPath[1] = '/'; outputPath[2] = '\0'; (*pcommMode) = 2; // reduce parameters


    printf("### Reading from %s starts. ###\n", confPath);


    lineNum = -1;
    while(fgets(line, LINE_MAX_LEN, fr) != NULL)
    {
        lineNum ++;

        sscanf (line, "%s", line2);
        if (strncmp(line2, "#", 1) == 0)
            continue;


        r = sscanf (line, "%s = %s", varname, varval);

        if (r <= 0)/* no input data (r == EOF) or no scanable data */
            continue;

        if (r < 2)
        {
            /* r = 0, empty line; r = 1, only one input (not acceptable when mode = 2); 2, two inputs */

            printf("Not enough input in line %d (starting from 0):\n%s", lineNum, line);
            printf("Failed to read file: %s\n", confPath );
            printf("Each line in conf.txt should be in format \"varname = varval\" or begin with \"#\" to ignore the line.\n");
            fclose(fr);
            return -2;
        }


        flag = 0;
        if (strcmp(varname, "netPath") == 0)
        {
            strcpy(netPath, varval);
            *commPathsN = 0;
            flag = 1;
        }
        if (strcmp(varname, "Kori") == 0)
        {
            sscanf (varval, "%d", pKori);
            printf("%s = %d\n", varname, *pKori);
        }
        if (strcmp(varname, "Kinit") == 0)
        {
            sscanf (varval, "%d", pKinit);
            printf("%s = %d\n", varname, *pKinit);
        }

        if (flag == 1)
            printf("%s = %s\n", varname, varval);






    }
    

    fclose(fr);


    printf("### Reading from %s finishes. ###\n", confPath);

    return 0;

}


/* a light function only to obtain the number of nodes */
extern int FnReadFromFileNetN(int rank, int readProc, char* netPath, int* pN, int* pNmin, MPI_Comm *pcommAll, int isPrintWarning)
{

    int N = 0, Nlocal;
    char line[LINE_MAX_LEN];
    int lineNum, lineId;
    int i, j, flag, procId, isSelfConn = 0;
    FILE *fr, *fw;
    int idxi, idxj, iTemp, iTemp2, Nmax, Nmin, isNoted = 0;




    if (rank == readProc)
    {

        /* first read N, and then read row and column configuration */
        /* declare the file pointer */
        fr = fopen (netPath, "rt");
        if (fr == NULL)
        {
            printf("Failed to open file: %s\n", netPath );
            return -1;
        }
        lineNum = 0;
        while(fgets(line, LINE_MAX_LEN, fr) != NULL)
        {

            if (sscanf (line, "%d %d", &idxi, &idxj) != 2)/* if not enough input in sscanf, values of the variables won't be changed. */
            {
                /* 0, empty line; 1, only one input (not acceptable); 2, two inputs */

                printf("Not enough input in line %d:\n%s", lineNum, line);
                printf("Failed to open file: %s\n", netPath );
                fclose(fr);
                return -2;
            }
            if (idxi < 0 || idxj < 0 )
            {
                printf("Wrong idx in line %d:\n%s", lineNum, line);
                printf ("idxi = %d, idxj = %d\n", idxi, idxj);

                printf("Failed to open file: %s\n", netPath );
                fclose(fr);
                return -2;

            }
            if (isPrintWarning == 1)
                if (idxi == idxj && isNoted == 0 )
                {
                    printf("Wrong idx in line %d:\n%s", lineNum, line);
                    printf ("idxi = %d, idxj = %d\n", idxi, idxj);
                    isNoted = 1;
                    printf("Warning: Maybe more. They are ignored in Opening file: %s\n", netPath );
                }


         

            if (lineNum == 0)
            {
                N = max(idxi, idxj);
                Nmin = min(idxi, idxj);
            }

            if (max(idxi, idxj) > N)
            {
                N = max(idxi, idxj); /* INT_MAX */
            }
            if (min(idxi, idxj) < Nmin)
            {
                Nmin = min(idxi, idxj); /* INT_MAX */
            }

            lineNum ++;
        }










      
    }
    


    N = N - Nmin + 1;
    MPI_Bcast(&N, 1, MPI_INT, readProc, *pcommAll);
    *pN = N;
    if (pNmin != NULL)
        *pNmin = Nmin;

    if (rank == readProc)
    {
        fclose(fr);  /* close the file prior to exiting the routine */

    }

    return 0;
}

extern int FnReadFromFileNet(int rank, int readProc, int outputrank, char* netPath, int* pN, int* plocalN, struct Mat* pWlocal, int size, MPI_Comm *pcommAll, int* databuffs, int** databuffr_c, int* datasizer_c, struct CommEle* pcomme, char* outputPath, char* path, int isReorder /* if 1, reorder the data for load balance */, int *nodeidNewOrder )
{

    int N = 0, Nlocal, direction;
    double timeT2;
    char line[LINE_MAX_LEN];
    int lineNum, lineId;
    int i, j, flag, procId;
    FILE *fr, *fw;
    int idxi, idxj, iTemp, iTemp2;
    int nodeStart, nodeEnd, nodeStartFul[size], nodeEndFul[size], *Wnnz,  comProcId;
    float vTemp = 1;
    MPI_Request *reqs, *reqr = (*pcomme).reqr;
    int **sbuff, **rbuff;
    int termFlagLocal, termFlag, convgFlag = 0, changeCounter = 0;
    int datasizes, isEmpBuff, datasizer, dataidxTemp;
    int *iptrTemp;
    int *WnnzFul, maxWnnzFul, range[2], binsize, binid;  /* reorder the data */
    struct binele
    {
        int n /* number of elements in the bin */;
        int *iarray /* idx array */, idx;
    } **binCounter, *pbinCounter;

    int ncols;
    const int *cols;
    const float *vals;
    int rRead, Nmin,   sendFreq, sendFreqidx, recvFreq, recvFreqIdx;




    rRead = FnReadFromFileNetN(rank, readProc, netPath, &N, &Nmin, pcommAll, 0);
    if (rRead < 0)
        return rRead;


    if (rank == readProc)
        fr = fopen (netPath, "rt");


    if (rank == readProc && isReorder == 1)
    {
        /* always allocate, because before, N is unknown */
        if (N > MAXN_FNORDER)
        {
            if (outputrank >= 0)
                printf("[%d] nodeidNewOrder is allocated. \n", rank);
            nodeidNewOrder = malloc(sizeof(int)*N);
        }


        WnnzFul = malloc(sizeof(int)*N);

        for ( i = 0; i<N; i++)
            WnnzFul[i] = 0;



        lineId = -1;
        rewind(fr);

        while(fgets(line, LINE_MAX_LEN, fr) != NULL)
        {
            lineId ++;  /* if (lineId > 1) break; printf("test on small file. "); */
            sscanf (line, "%d %d", &idxi, &idxj);




            for (j = 0; j<2; j++)
            {
                if(j==0)
                    /* only consider undirected & unweighted graphs */
                    /* it could be a symmetric graph while the data file only has upper/lower triangular */
                {
                    iTemp = idxi;
                    iTemp2 = idxj;
                }
                if(j==1)
                {
                    iTemp = idxj;
                    iTemp2 = idxi;
                }

                /* -Nmin: convert to zero based index */
                iTemp = iTemp - Nmin;
                iTemp2 = iTemp2 - Nmin;



                WnnzFul[iTemp] ++;




            }

        }

        range[0] = N+1;
        range[1] = -1;

        for (i = 0; i<N; i++)
        {
            if(WnnzFul[i] < range[0])
                range[0] = WnnzFul[i];

            if(WnnzFul[i] > range[1])
                range[1] = WnnzFul[i];
        }

        binsize = range[1] - range[0] + 1; /* consider each value as a bin; bin size may be reduced by grouping nearby values (set up a specific bin number) */
        binCounter = malloc(sizeof(struct binele *)*(binsize)); /* may use sparse matrix here */

        for (i = 0; i<binsize; i++)
        {
            binCounter[i] = NULL;
        }

        for (i = 0; i<N; i++) /* compute the size for each bin */
        {
            binid = WnnzFul[i] - range[0];
            if (binCounter[binid] == NULL)
            {
                binCounter[binid] = malloc(sizeof(struct binele));
                (*(binCounter[binid])).n = 0;
                (*(binCounter[binid])).idx = 0;
            }
            ((*(binCounter[binid])).n) ++;
        }


        for (binid = 0; binid<binsize; binid++) /* allocate memory for each bin */
        {
            if (binCounter[binid] != NULL)
            {
                (*(binCounter[binid])).iarray = malloc(sizeof(int)*(*(binCounter[binid])).n);
            }
        }


        for (i = 0; i<N; i++) /* put all the node ids into bins according to their WnnzFul */
        {
            binid = WnnzFul[i] - range[0];
            pbinCounter = binCounter[binid];
            (*pbinCounter).iarray[(*pbinCounter).idx] = i;
            ((*pbinCounter).idx) ++;


        }

        /* compute the starting idx of each processor */
        for (i = 0; i<size; i++)
        {
            FnNodeRange(N, size, i, &(nodeStartFul[i]), &(nodeEndFul[i]));

        }


        /* set the new index for balancing the edges in each proc */
        procId = 0;
        direction = 0;
    
        iTemp = 0;
        for (i = 0; i<binsize; i++)
        {
            pbinCounter = binCounter[i];
            if (pbinCounter != NULL)
            {
                for (j = 0; j<(*pbinCounter).n; j++)
                {
                    while (1)
                    {
                        if (nodeStartFul[procId] < nodeEndFul[procId])
                            break;
                        else
                        {
                            /* assume the node number is much larger than the MPI ranks, so the way below is more balanced */
                            if (direction == 0)
                            {
                                procId ++;
                                if (procId >= size)
                                {
                                    procId = size-1;
                                    direction = 1;
                                }
                            }
                            else
                            {
                                procId --;
                                if (procId < 0)
                                {
                                    procId = 0;
                                    direction = 0;
                                }

                            }

                        }

                    }
                    iTemp ++;
                
                    nodeidNewOrder[(*pbinCounter).iarray[j]] = nodeStartFul[procId];
                    (nodeStartFul[procId]) ++;

                    /* assume the node number is much larger than the MPI ranks, so the way below is more balanced */
                    if (direction == 0)
                    {
                        procId ++;
                        if (procId >= size)
                        {
                            procId = size-1;
                            direction = 1;
                        }
                    }
                    else
                    {
                        procId --;
                        if (procId < 0)
                        {
                            procId = 0;
                            direction = 0;
                        }

                    }


                }


            }


        }

        if (N > MAXN_FNORDER)
        {
            strcpy( path, outputPath );
            strcat( path, "nodeidNewOrder.tmp"); /* in this file, each line representing the new index of the node whose id corresponds to the lineid */
            fw = fopen (path, "w");
            if (fw == NULL)
            {
                printf("Failed to open file to write (maybe wrong folder): %s\n", path );
                return -1;
            }
            for (i = 0; i<N; i++)
            {
                fprintf(fw, "%d\n", nodeidNewOrder[i]);
                
            }
           
            fclose(fw);
        }


    }








    FnNodeRange(N, size, rank, &nodeStart, &nodeEnd);
    Nlocal = nodeEnd - nodeStart;
    Wnnz = malloc(sizeof(int)*Nlocal);
    for (i = 0; i<Nlocal; i++)
    {
        Wnnz[i] = 0;

    }




    flag = 0;



    for(i=0; i<2; i++)/* send twice: one for matcreate, and one for matsetvalues */
    {

        lineId = -1;

        if (i == 1)
        {
            MPI_Barrier(*pcommAll); /* make sure everyone arrives here to setup the matrix */

            for (j = 0; j<Nlocal; j++)
            {



                if (Wnnz[j] == 0)
                {


                    if (flag != 1)
                        printf("[%d] wnnz0 = (local) %d (global) %d (start) %d (end) %d (Nlocal) %d \n", rank,  j, j + nodeStart, nodeStart, nodeEnd, Nlocal);
                    flag = 1;
                    
                }
              

            }

            MatCreateT(pWlocal, N, N, nodeStart, nodeEnd, Wnnz, 1, *pcommAll);
        }

        sendFreqidx = 0;
        sendFreq = PSENDRECV_BUFF /* default: PSENDRECV_BUFF*/;
        
        if (rank == readProc)
        {



            rewind(fr);
            FnSendMessageStart(&isEmpBuff, pcomme);

            while(fgets(line, LINE_MAX_LEN, fr) != NULL)
            {
                lineId ++;  /* if (lineId > 1) break; printf("test on small file. "); */
                sscanf (line, "%d %d", &idxi, &idxj);


                for (j = 0; j<2; j++)
                {
                    if(j==0)
                        /* only consider undirected & unweighted graphs */
                        /* it could be a symmetric graph while the data file only has upper/lower triangular */
                    {
                        iTemp = idxi;
                        iTemp2 = idxj;
                    }
                    if(j==1)
                    {
                        iTemp = idxj;
                        iTemp2 = idxi;
                    }

                    /* -Nmin: convert to zero based index */
                    iTemp = iTemp - Nmin;
                    iTemp2 = iTemp2 - Nmin;


        
                    if (isReorder == 1)
                    {
                        iTemp = nodeidNewOrder[iTemp];
                        iTemp2 = nodeidNewOrder[iTemp2];
                    }



                    FnComProcId(iTemp, &comProcId, N, size);
                    if (comProcId != readProc) /* don't send message to it self */
                    {





                        databuffs[0] = iTemp;
                        databuffs[1] = iTemp2;
                        datasizes = 2;
                        while(1)
                        {
                            /* always hold until it's full */
                            FnSendMessageBuff(&isEmpBuff, &changeCounter, databuffs, datasizes, &comProcId, 1, rank, pcomme);
                            if (isFreqTrue(&sendFreqidx, &sendFreq))
                                FnSendMessageNowProf(rank, pcomme, &timeT2) ;
                            if (isEmpBuff == 1)
                                break;
                        }

                    }
                    else
                    {
                        if (i==0)
                            Wnnz[iTemp - nodeStart] ++;

                        if (i==1)
                        {
                            vTemp = 1;


                            
                            if (MatSetValueT(pWlocal,iTemp,iTemp2, 1)<0)
                                return -1;

                        }
                    }


                }
                
            }




            while(1)
            {
                /* MPI_Isend(NULL, 0, MPI_INT, 1, 0, MPI_COMM_WORLD, &((*pcomme).reqs[1]));printf("EE2\n");*/
                FnSendMessageTerm(&isEmpBuff, &termFlagLocal, &termFlag, &convgFlag, i, changeCounter, rank, size, pcomme); /* tag should not be 0 */

                if (isEmpBuff == 1)
                    break;
            }

        }

        if (rank != readProc)
        {

            
            iTemp = readProc;
            FnRecvMessageStartSome(rank, &iTemp , 1, pcomme);


            isEmpBuff = 1;
            termFlag = -1;
            while(1)
            {




                if (FnRecvMessageProf(&termFlag, &convgFlag, rank, &datasizer, databuffr_c, datasizer_c, pcomme, &timeT2) == 1)
                {


                    for (iTemp = 0; iTemp < datasizer; iTemp++)
                    {
                        dataidxTemp = 0;
                        iptrTemp = databuffr_c[iTemp];
                        while(dataidxTemp < datasizer_c[iTemp])
                        {
                            


                            if (i==0)
                            {
                                Wnnz[iptrTemp[dataidxTemp] - nodeStart] ++;

                                
                            }

                            if (i==1)
                            {
                                vTemp = 1;


                                
                                if (MatSetValueT(pWlocal, iptrTemp[dataidxTemp], iptrTemp[dataidxTemp+1], 1)<0)
                                    return -1;
                            }

                            dataidxTemp += 2; /* assume totally 3 ints */


                        }

                    }


                }
                if (termFlag == 0)
                    break;





            }

        }
    }


    if (rank == readProc)
    {
        fclose(fr);  


    }





    MatAssemble(pWlocal);
   




    if (rank == readProc && isReorder == 1)
    {
        if (N > MAXN_FNORDER)
        {
            if (outputrank >= 0)
                printf("[%d] nodeidNewOrder is freed. \n", rank);
            free(nodeidNewOrder);
        }
        free(WnnzFul);
        for (i = 0; i<binsize; i++)
        {
            if (binCounter[i] != NULL)
            {
                free((*(binCounter[i])).iarray);
                free(binCounter[i]);
            }
        }
        free(binCounter);
    }
    free(Wnnz);


    if (flag == 1)
        printf("Warning: Isolated nodes exist in the graph. \n");/* need to fix correspondingly the update Z function */




    *pN = N;
    *plocalN = Nlocal;



    return 0;

}



/* "oton": when pZidxFul is in the original order, use this function to convert it to the new order to match the adjacency matrix in the memory (new order). The matrix in the file is of original order. */
/* "ntoo": when pZidxFul is in the new order, convert the output result to the original order. Can be parallelized by scattering nodeidNewOrder to and collecting ZidxFul from other procs */
extern int FnConvertOrder(int N, int **pZidxFul, char* outputPath, char* path, int isReorder, int rank, int readProc, int outputrank,  const char *convertType, int *nodeidNewOrder)
/* convertType: "oton", original to new; "ntoo", new to original */
{
    FILE *fr;
    int *ZidxFulNew, *ptr, lineId, i;
    int *ZidxFul = *pZidxFul;
    char line[LINE_MAX_LEN], convertTypeId = 0;
    if (isReorder == 0 || rank != readProc)
        return 0;


    if (strcmp(convertType, "oton") == 0)
        convertTypeId = 1;

    if (strcmp(convertType, "ntoo") == 0)
        convertTypeId = 2;

    if (convertTypeId == 0)
    {
        printf("Wrong convert type in FnConvertOrder(...).");
        return -1;
    }



    if (N > MAXN_FNORDER)
    {
        if (outputrank >= 0)
            printf("[%d] nodeidNewOrder is allocated. \n", rank);
        nodeidNewOrder = malloc(sizeof(int)*N);

        strcpy( path, outputPath );
        strcat( path, "nodeidNewOrder.tmp"); 
        fr = fopen (path, "rt");

        lineId = -1;

        while(fgets(line, LINE_MAX_LEN, fr) != NULL)
        {
            lineId ++;  
            sscanf (line, "%d\n", &(nodeidNewOrder[lineId]));

        }

        if (lineId != N - 1)
        {
            printf("Wrong number of input lines (%d). ", lineId);
            printf("Failed to open file: %s", path );
            return -1;
        }

    }


    if (convertTypeId == 2)  /* convert from original to new order */
    {
        ZidxFulNew = malloc(sizeof(int)*N);
        for (i = 0; i<N; i++)
            ZidxFulNew[i] = ZidxFul[nodeidNewOrder[i]];
        ptr = ZidxFul;
        ZidxFul = ZidxFulNew;
        ZidxFulNew = ptr;
        free(ZidxFulNew);
    }

    if (convertTypeId == 1) /* it's a convert */
    {
        ZidxFulNew = malloc(sizeof(int)*N);
        for (i = 0; i<N; i++)
            ZidxFulNew[nodeidNewOrder[i]] = ZidxFul[i];
        ptr = ZidxFul;
        ZidxFul = ZidxFulNew;
        ZidxFulNew = ptr;
        free(ZidxFulNew);



    }

    *pZidxFul = ZidxFul;

  
    if (N > MAXN_FNORDER)
    {
        if (outputrank >= 0)
            printf("[%d] nodeidNewOrder is freed. \n", rank);
        free(nodeidNewOrder);
    }
    return 0;

}







extern int FnSendMessageStart(int *isEmpBuff, struct CommEle* pcomme)
{
    int i, iTemp;

    /* all the sending buffers are clear */
    *isEmpBuff = 1;

    for (i = 0; i<(*pcomme).ncomm; i++) /* *2 corresponds to FnPersistentCommInit */
        (*pcomme).sIdx[i] = PSENDRECV_START;


    /* wait for unfinished send */
    MPI_Waitall((*pcomme).ncomm, (*pcomme).reqs, MPI_STATUSES_IGNORE);




    return 0;
}




/* only write to buffer */
extern int	FnSendMessageBuff(int *pisEmpBuff, int* pchangeCounter, int* databuff, int datasize, int* dest, int destNum, int rank, struct CommEle* pcomme)
{
    int i, j, flag, isFirstTry, isBuffNowRoom;
    int iTemp, buffidTemp, *unsent = (*pcomme).unsentNode, *pbuffNow, *pbuffAlt, *psIdx;
    MPI_Request* reqs;
    int** sbuff;

    reqs = (*pcomme).reqs;
    sbuff = (*pcomme).sbuff;

    isFirstTry = *pisEmpBuff; /* all the buffers are empty, so it's the first try to send those databuff to all the processors */
    *pisEmpBuff = 1;
    for (j = 0; j<destNum; j++)
    {

        iTemp = dest[j];/* destination processor */
        if (iTemp == rank)
            continue;



        pbuffNow = &((*pcomme).sbuffNow[iTemp]);
        pbuffAlt = &((*pcomme).sbuffAlt[iTemp]);
        psIdx =  &((*pcomme).sIdx[iTemp]);

       

        if (isFirstTry || unsent[iTemp] != 0)/* if first sending, or failure in previous sending, try to send; otherwise, do nothing. */
        {



            isBuffNowRoom = 1;
            if (*psIdx + datasize + PSENDRECV_TERM > PSENDRECV_BUFF)
                isBuffNowRoom = 0;





         
            if (isBuffNowRoom == 0)
            {
                MPI_Test(&(reqs[*pbuffAlt]), &flag, MPI_STATUS_IGNORE);
                if (flag == 0)/* if the alternative buff has not been sent, and the current buff is full, don't send */
                {
                    unsent[iTemp] = 1;
                    *pisEmpBuff = 0;
                    continue;
                }
                else
                {
                    sbuff[*pbuffNow][0] = (*psIdx);/* corresponding to PSENDRECV_START*/
                    /* MPI_Start(&(reqs[*pbuffNow]));*/
                    MPI_Isend(sbuff[*pbuffNow], sbuff[*pbuffNow][0], MPI_INT, dest[j], 0, MPI_COMM_WORLD, &(reqs[*pbuffNow]));
                  

                    *psIdx = PSENDRECV_START; /* restore index after data sent */
                    buffidTemp = *pbuffNow;
                    *pbuffNow = *pbuffAlt;
                    *pbuffAlt = buffidTemp;



                    if (*psIdx + datasize + PSENDRECV_TERM < PSENDRECV_BUFF)
                        isBuffNowRoom = 1;
                    else
                    {
                        /* not enough space even for new buffer */
                        isBuffNowRoom = 0;
                        printf("Error: Buffer for MPI_Isend is too small.\n");
                        return 1;
                    }



                }
            }


            if (isBuffNowRoom == 1)
            {

                memcpy(&(sbuff[*pbuffNow][*psIdx]), databuff, datasize*sizeof(int));
                *psIdx = *psIdx + datasize;
                unsent[iTemp] = 0;


                


            }









        }


    }

   
    if (isFirstTry == 1)
        if (pchangeCounter != NULL)
            (*pchangeCounter) ++;

    return 0;
}


/* send all the buffer now */
extern int	FnSendMessageNowProf(int rank, struct CommEle* pcomme, double *ptime)
{

    int i, flag;
    float timeT;
    int buffidTemp, *pbuffNow, *pbuffAlt, *psIdx;
    MPI_Request* reqs;
    int** sbuff;

    reqs = (*pcomme).reqs;
    sbuff = (*pcomme).sbuff;

    *ptime = 0;


    for ( i = 0; i < (*pcomme).size; i++)
    {
        if ((*pcomme).sIdx[i] <= PSENDRECV_START) /* if no data, don't send the buff */
            continue;

        pbuffNow = &((*pcomme).sbuffNow[i]);
        pbuffAlt = &((*pcomme).sbuffAlt[i]);
        psIdx =  &((*pcomme).sIdx[i]);

        /*timeT = MPI_Wtime();*/
        MPI_Test(&(reqs[*pbuffAlt]), &flag, MPI_STATUS_IGNORE);
        if (flag == 0)
            continue;

        *ptime += 1;

        sbuff[*pbuffNow][0] = *psIdx;/* corresponding to PSENDRECV_START*/
        MPI_Isend(sbuff[*pbuffNow], sbuff[*pbuffNow][0], MPI_INT, i, 0, MPI_COMM_WORLD, &(reqs[*pbuffNow]));




        *psIdx = PSENDRECV_START; /* restore index after data sent */
        buffidTemp = *pbuffNow;
        *pbuffNow = *pbuffAlt;
        *pbuffAlt = buffidTemp;

    }


    return 0;

}



extern int	FnSendMessageTerm(int *pisEmpBuff, int* ptermFlagLocal, int* ptermFlag, int* pconvgFlag,  int tag, int changeCounter, int rank, int size, struct CommEle* pcomme)
{
    int j, isTermSent;
    MPI_Request* reqs;
    int** sbuff, *pbuffNow,  *psIdx;

    reqs = (*pcomme).reqs;
    sbuff = (*pcomme).sbuff;


    *pisEmpBuff = 1;
    for (j = 0; j < size; j++ )
    {
        if (j == rank)
        {
            continue;
        }

        pbuffNow = &((*pcomme).sbuffNow[j]);
        psIdx =  &((*pcomme).sIdx[j]);
        isTermSent = 0;
        if (*psIdx >= PSENDRECV_TERM) /* seek for previously written data */
            if (sbuff[*pbuffNow][*psIdx - PSENDRECV_TERM] == -1 ) /* no need to check the tag */
                isTermSent = 1;

        if (isTermSent == 0)
        {



            (*psIdx) += PSENDRECV_TERM;/* add terminating code */
            sbuff[*pbuffNow][*psIdx-PSENDRECV_TERM] = -1;
            sbuff[*pbuffNow][*psIdx-PSENDRECV_TERM+1] = changeCounter;
            sbuff[*pbuffNow][*psIdx-PSENDRECV_TERM+2] = tag;
            /* corresponding to PSENDRECV_START*/
            sbuff[*pbuffNow][0] = (*psIdx);
            MPI_Isend(sbuff[*pbuffNow], sbuff[*pbuffNow][0], MPI_INT, j, 0, MPI_COMM_WORLD, &(reqs[*pbuffNow]));
        }



    }

    if (*pisEmpBuff == 1)
    {
        (*pconvgFlag) += changeCounter;
        *ptermFlagLocal = 0;
        (*ptermFlag) ++;/* note: ++ has a higher priority than * */
    }


    return 0;
}


extern int FnRecvMessageStart(int rank, int size, struct CommEle* pcomme)
{

    int i, iTemp;
    MPI_Request* reqr = (*pcomme).reqr;


    /* wait for unfinished recv */
    MPI_Waitall((*pcomme).ncomm, (*pcomme).reqr, MPI_STATUSES_IGNORE);

    for (i=0; i<size; i++)
        if (i != rank)
        {
           
            iTemp = (*pcomme).rbuffNow[i];
            MPI_Irecv((*pcomme).rbuff[iTemp], PSENDRECV_BUFF, MPI_INT, i, 0, MPI_COMM_WORLD, &(reqr[iTemp]));


        }



    return 0;
}


extern int FnRecvMessageStartSome(int rank, int *sourceList , int sourceCount, struct CommEle* pcomme)
{
    /* when start, pdataloc in FnRecvMessage may change, and should not use */

    int i, iTemp;
    MPI_Request* reqr = (*pcomme).reqr;


    /* wait for unfinished recv */
    MPI_Waitall((*pcomme).ncomm, (*pcomme).reqr, MPI_STATUSES_IGNORE);

    for (i=0; i<sourceCount; i++)
        if (sourceList[i] != rank)
        {
            iTemp = (*pcomme).rbuffNow[sourceList[i]]; 
            MPI_Irecv((*pcomme).rbuff[iTemp], PSENDRECV_BUFF, MPI_INT, sourceList[i], 0, MPI_COMM_WORLD, &(reqr[iTemp]));


        }



    return 0;
}




extern int FnRecvMessageProf(int* ptermFlag, int* pconvgFlag, int rank, int* pdatasizer, int** pdataloc_c, int* pdatasize_c, struct CommEle* pcomme, double *ptime)
{
    int i, j, isTerm, rValue = -1;
    int* rbuff, *pbuffNow, *pbuffAlt, buffidTemp, source;
    MPI_Request* reqr;



    *ptime = MPI_Wtime();

    rValue = 0;
    
    
    MPI_Testsome((*pcomme).ncomm, (*pcomme).reqr, &((*pcomme).countr), (*pcomme).reqr_com, MPI_STATUS_IGNORE); /* may only test the active portion for speed */

    *ptime = MPI_Wtime() - *ptime;

    *pdatasizer = (*pcomme).countr;



    if ((*pcomme).countr == MPI_UNDEFINED)
    {

        return 0;
    }

    for (i = 0; i < (*pcomme).countr; i++)
    {
        /* set (*pcomme).countr = (*pcomme).size and test here ???? */
        source = ((*pcomme).reqr_com[i])%((*pcomme).size);
        pbuffNow = &((*pcomme).rbuffNow[source]);/* note: pbuffNow == (*pcomme).reqr_com[i] */
        pbuffAlt = &((*pcomme).rbuffAlt[source]);
        if (*pbuffNow != (*pcomme).reqr_com[i]) /* it's not the currently active buffer, skip for efficiency. ???? no need to check  */
            continue;


        rbuff = (*pcomme).rbuff[*pbuffNow];
        reqr = &((*pcomme).reqr[*pbuffNow]);


   






        pdatasize_c[i] = rbuff[0];
        isTerm = 0;



    

        /* check termination */
        if ((pdatasize_c[i]) >= PSENDRECV_START + PSENDRECV_TERM)
        {

            if (rbuff[pdatasize_c[i] - PSENDRECV_TERM] == -3 )
            {
                pdatasize_c[i] = 0;
                continue; /* this message has been processed and already terminated */
            }


            if (rbuff[pdatasize_c[i] - PSENDRECV_TERM] == -1 )
            {
                (*ptermFlag) ++;
                (*pconvgFlag) += rbuff[pdatasize_c[i] - PSENDRECV_TERM + 1];
                rbuff[pdatasize_c[i] - PSENDRECV_TERM] = -3; /*mark as read, no more receiver needed for this source */
                (pdatasize_c[i]) -= PSENDRECV_TERM; /* the termination data are processed */
                isTerm = 1;


            }


        }




        pdatasize_c[i] = pdatasize_c[i] - PSENDRECV_START;
        if ((pdatasize_c[i]) > 0) /* still some data left */
        {
            pdataloc_c[i] = &(rbuff[PSENDRECV_START]);
            rValue = 1;
        }

        /* if not terminated, switch channel and start the new channel */
        if (isTerm == 0)
        {
            buffidTemp = *pbuffNow;
            *pbuffNow = *pbuffAlt;
            *pbuffAlt = buffidTemp;

            reqr = &((*pcomme).reqr[*pbuffNow]);
            /* MPI_Start(reqr);*/
            MPI_Irecv((*pcomme).rbuff[*pbuffNow], PSENDRECV_BUFF, MPI_INT, source, 0, MPI_COMM_WORLD, reqr);


        }


  
    }

    return rValue;

}




int FnCmpfunc (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}


extern int FnUniqueStart(struct PreLocSpace *pls, int nMax)
{
    (*pls).obj = malloc(sizeof(float)*nMax);
    (*pls).count = malloc(sizeof(int)*nMax);
    (*pls).countnz = malloc(sizeof(int)*nMax);
    (*pls).countnnz = nMax;

    return 0;

}


/* if "values" is not NULL, FnUniqueStart and FnUniqueTerm must be called */
extern int FnUnique(int *vec, int ncols, int *ncolsNew, float *values, struct PreLocSpace *pls)
{
    /* The output result is sorted. Assume elements in vec is non-negative */

    int colsIdx = -1, ele = -1;
    int i, *orderIdx, *unqvec;
    float *unqvalues;


    if (values == NULL)
    {
        qsort(vec, ncols, sizeof(int), FnCmpfunc);
        for (i = 0; i<ncols; i++)
        {
            if (vec[i] != ele)
            {
                colsIdx ++;
                ele = vec[i];
                vec[colsIdx] = vec[i];
            }
        }
        *ncolsNew = colsIdx+1;
    }

    if (values != NULL)
    {
        if ((*pls).countnnz < ncols)
        {
            printf("Not enough temporary space (%d < %d) in (*pls).countnz in FnUnique. May also check FnUniqueStart.\n",  (*pls).countnnz,ncols);
            exit(0);
        }
     

        orderIdx = (*pls).countnz;
        unqvalues = (*pls).obj;
        unqvec = (*pls).count;

        for (i = 0; i < ncols; i++)
        {
            orderIdx[i] = i;
            unqvalues[i] = 0;
        }


        sortint_index(ncols, vec, orderIdx, 0);




        for (i = 0; i<ncols; i++)
        {
            if (vec[orderIdx[i]] != ele)
            {
                colsIdx ++;
                ele = vec[orderIdx[i]];
                unqvec[colsIdx] = vec[orderIdx[i]];
            }

            unqvalues[colsIdx] += values[orderIdx[i]];

        }
        *ncolsNew = colsIdx+1;



        for (i = 0; i<*ncolsNew; i++)
        {
            vec[i] = unqvec[i];
            values[i] = unqvalues[i];
        }

    }




    return 0;

}


extern int FnUniqueTerm(struct PreLocSpace *pls)
{
    free((*pls).obj);
    free((*pls).count);
    free((*pls).countnz);
    (*pls).countnnz = 0;

    return 0;

}




extern int myGather(void *vec, int *dispgather, int *ngather, int rank, MPI_Datatype datatype, MPI_Comm * comm)
{


    if (datatype == MPI_INT)
        MPI_Allgatherv(& ((int*) vec)[dispgather[rank]], ngather[rank], datatype, vec, ngather, dispgather, datatype, *comm);
    else if (datatype == MPI_FLOAT)
        MPI_Allgatherv(& ((float*) vec)[dispgather[rank]], ngather[rank], datatype, vec, ngather, dispgather, datatype, *comm);
    else
        printf("Error: Undefined data type in function myGather(...).");

    return 0;

}


extern int myGatherLocal(void *vecSendLocal, void *vec, int *dispgather, int *ngather, int rank, MPI_Datatype datatype, MPI_Comm * comm)
{


    if (datatype == MPI_INT)
        MPI_Allgatherv(vecSendLocal, ngather[rank], datatype, vec, ngather, dispgather, datatype, *comm);
    else if (datatype == MPI_FLOAT)
        MPI_Allgatherv(vecSendLocal, ngather[rank], datatype, vec, ngather, dispgather, datatype, *comm);
    else
        printf("Error: Undefined data type in function myGatherLocal(...).");

    return 0;

}



extern int PrintArrayI(const int size, const int *ar)
{
    int i, rank;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    printf("[%d]. ", rank);

    for (i = 0; i<size; i++)
        printf("%d ", ar[i] );

    printf("\n");

    return 0;

}

extern int PrintArrayR( const int size, const float *ar)
{
    int i, rank;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    printf("[%d]. ", rank);

    for (i = 0; i<size; i++)
        printf("%f ", ar[i] );

    printf("\n");

    return 0;


}

/* Mat is a pointer, so passing this value is cheap
 http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/Mat.html#Mat
 Mat B;

 if (mode == 0), compute the whole Barray
 if (mode == 1), compute the Barray except Barray[K]
 if (mode == 2), compute only Barray[K]
 */

extern int UpdateB(float *Barray, float dampB, int* Nc, int *Ni, int rank, int K, int N, int totalEdges, struct PreLocSpace *ppls, int IstartB, int IendB, int mode, float *qout)
{


    float  *p = (*ppls).obj, q, Nis, Ncss;
    int i;

  
  
    if (mode < 0 || mode > 2)
    {
        printf("Error: Wrong mode selection in UpdateB. \n");
        exit(0);
    }



    if (mode == 0 || mode == 1)
    {

        Nis = 0;
        for (i=IstartB; i<IendB; i++)
        {
            p[i] = Ni[i]/(pow(Nc[i],2) + FLT_EPSILON);
            Nis += Ni[i];
        }

        Ncss = 0;
        for (i = IstartB; i<IendB; i++)
            Ncss += pow(Nc[i],2);


        for (i = IstartB; i<IendB; i++)
            Barray[i] = dampB*Barray[i] + (1-dampB)*p[i];
    }
  


    if (mode == 2)
    {
        Nis = p[0];
        Ncss = p[1];
    }

    if (mode == 0 || mode == 2)
    {
        q = (totalEdges*2 - Nis)/(pow(N,2) - Ncss + FLT_EPSILON);

        Barray[K] = dampB*Barray[K] + (1-dampB)*q*BKINIT; 
    }

    if (mode == 1)
    {
        p[0] = Nis;
        p[1] = Ncss;
    }

    if (mode == 2)
    {
      if (qout == NULL)
      {
        printf("Error: Memory for qout undefined. \n");
        return 0;
      }
      
      qout[0] = p[0]/p[1];
      qout[1] = q;
    }


    return 0;
}





void random_number_init(unsigned int seed)
{
    int rank;

    if (seed == -1)
        seed = time(NULL);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    seed = (unsigned int)rank*10+seed;
    srand(seed);
}


/* generate unsigned int, from 0, 1, 2, ..., rMax*/
int random_number_int( int rMax)
{




    if (RAND_MAX < rMax)
    {
        printf("Warning: RANDMAX (%d) is less than the maximum number in need (%d).\n ", RAND_MAX, rMax);
    }


    return floor(random_number_float()*(rMax+1));

}

float random_number_float() /* f is [0,1) */
{
    /* rand is 0 to RAND_MAX. */
    float val;

    do
    {
        val = rand()/(float)(RAND_MAX); /* note: RAND_MAX+1 will overflow */
    }
    while(val == 1);

    return val;
}




int max(int a, int b)
{
    return (a > b) ? a : b;
}


int min(int a, int b)
{
    return (a < b) ? a : b;
}


float minfloat(float a, float b)
{
    return (a < b) ? a : b;
}


int isFreqTrue(int* pFreqIdx, int* pFreq)
{


    (*pFreqIdx) ++;
    if (*pFreqIdx >= *pFreq) /* if meet the frequency, go! */
    {
        *pFreqIdx = 0;
        return 1;
    }
    return 0;

}


/* output: orderIdx (only size Kori needed) */
/* do it in parallel: build a histogram of Nc values, and pick the appropriate criteria, finally remove those Nc less than certain value  */
int FnTopNcIndices(int K, int Kori, int *Nc, int *orderIdx, int IstartB, int IendB, int NlocalB, float *Barray, float *Barray2, struct PreLocSpace *pls, int rank, MPI_Comm *pcommAll)
{


    int flag, isUpdateBin, i, j, isUseExtra;
    int *NcLocal = (*pls).arrNlocalB, threshold /* valid values should be larger than the threshold */, extra, range[2] /* min (exclude) and max (include) values in the range */, rangeR[2], rangeIdx[2] /* beginning (include) and ending (include) index in NcLocal contained by range */, orderIdxi, binPrev, binCountCum[BINNUM]/* the cumulative count of elements in each bin (corresponding to binIdx and binVal); cumulates from the end to the beginning  */, binCountCumR[BINNUM], bini;

    struct bin
    {
        int max, min /* value x in the bin are min< x <= max */, maxIdx, minIdx  /* the corresponding index of the minimum and maximum values of Nc in the bin */;
        int count;
    } bins[BINNUM]; /* bins[i] indicate the i-th bin  */


    for (j = IstartB; j < IendB; j++)
        NcLocal[j-IstartB] = Nc[j]; /* may use the space for NcLocal */

    

    qsort(NcLocal, NlocalB, sizeof(int), FnCmpfunc);  /* ascending order */
    

    binPrev = 0;
    range[0] = -(NcLocal[0] - 1); /* exclude the value of range[0], so it should be one less */
    range[1] = NcLocal[NlocalB-1];
    MPI_Allreduce(range, rangeR, 2, MPI_INT, MPI_MAX, *pcommAll);
    for (j = 0; j<2; j++)
        range[j] = rangeR[j];
    range[0] = -range[0]; /* so the boundary is between the min(Nc)-1 (exclude) and max(Nc) */
    rangeIdx[0] = 0;
    rangeIdx[1] = NlocalB - 1;

  

    while(1)
    {

        for (i = 0; i < BINNUM; i ++)
        {
            bins[i].max = range[0] + (i+1) /((float)BINNUM)*(range[1] - range[0]);
            bins[i].min = range[0] + i /((float)BINNUM)*(range[1] - range[0]);
            bins[i].maxIdx = -1;
            bins[i].minIdx = -1;
            bins[i].count = 0;
        }
    
        bini = BINNUM - 1;
        if (rangeIdx[1] != -1) /* when the count is 0 for a bin, the extended rangeIdx will be [-1, -1] */
            for (i = rangeIdx[1]; i >= rangeIdx[0]; i--)
            {


                isUpdateBin = 0;
                if (NcLocal[i] > bins[bini].min && NcLocal[i] <= bins[bini].max)
                {
                    isUpdateBin = 1;
                }
                else
                {
                    while(1) /* seek for the next bin */
                    {

                        bini --;
                        if (bini < 0)
                            printf("Something Wrong in function TopNcIndices(). \n");
                        if (NcLocal[i] > bins[bini].min && NcLocal[i] <= bins[bini].max)
                        {
                            isUpdateBin = 1; /* found the right bin, so assign */
                            break;
                        }

                    }
                }

                if (isUpdateBin == 1)
                {
                    (bins[bini].count) ++;
                    if (bins[bini].maxIdx == -1)
                        bins[bini].maxIdx = i;

                    bins[bini].minIdx = i; /* because NcLocal[i] is ordered from min to max */
                }





            }




        binCountCum[BINNUM-1] = bins[BINNUM-1].count;
        for (i = BINNUM-2; i>=0; i--)
        {
            binCountCum[i] = binCountCum[i+1] + bins[i].count;

        }



        MPI_Allreduce(binCountCum, binCountCumR, BINNUM, MPI_INT, MPI_SUM,  *pcommAll);
        for (j = 0; j<BINNUM; j++)
            binCountCum[j] = binCountCumR[j];



        flag = 0;
        for (i = BINNUM - 1; i >= 0; i--)
        {
            if (binPrev + binCountCum[i] == Kori )
            {
                threshold = bins[i].min; /* valid values should be larger than the threshold */
                extra = 0;
                flag = 1;
                break;

            }


            if (binPrev + binCountCum[i] > Kori)
            {

                range[0] = bins[i].min;
                range[1] = bins[i].max;
                rangeIdx[0] = bins[i].minIdx;
                rangeIdx[1] = bins[i].maxIdx;

                if (i < BINNUM - 1) /* add all the cumulative count larger than the current range */
                    binPrev = binPrev + binCountCum[i + 1];

                flag = 2;
                break;
            }



        }


        if (flag == 1)
            break;


        if (flag == 2)
            if (range[1] - range[0] <= 1)/* identify the number of extra nodes */
            {
                threshold = range[0];
                extra = Kori - binPrev; /* the extra number of Nc entries needed at (threshold+1); there are more entries than the extra number at the threshold. */
                break;
            }


    }


    orderIdxi = 0;
    isUseExtra = 0;
    if (extra>0)
        isUseExtra = 1;
    for (i = 0; i<K; i++)
    {

        if (Nc[i] > threshold)
        {
            if (isUseExtra == 1) 
                if (Nc[i] == threshold + 1)
                {
                    if (extra == 0)
                        continue;
                    else
                        extra --;
                }



            orderIdx[orderIdxi] = i;
            orderIdxi ++;
        }


    }




    return 0;
}


/* N1, global size of the first dimension
 N2, global size of the second dimension
 rowIdxStart, the starting index of the local row
 rowIdxEnd, one more than the ending index of the local row (the actual ending index is rowIdxEnd - 1)
 Wnnz, each entry indicate the maximum number of none zeros in the corresponding local row */
int MatCreateT(struct Mat *pWlocal, int N1, int N2, int rowIdxStart, int rowIdxEnd, int *Wnnz, int isBooleanMat, MPI_Comm comm)
{
    int Nlocal, j;

    if (isBooleanMat < 0 || isBooleanMat > 1)
    {
        printf("Wrong parameter setting in MatCreateT: isBooleanMat = %d, which must be 0 or 1.\n", isBooleanMat);
        exit(0);
    }
    (*pWlocal).isBooleanMat = isBooleanMat;
    (*pWlocal).comm = comm;

    (*pWlocal).size[0] = N1; /* MatSetSizes(*pWlocal,Nlocal,Nlocal,N,N); */
    (*pWlocal).size[1] = N2;
    (*pWlocal).Istart = rowIdxStart;
    (*pWlocal).Iend = rowIdxEnd;
    Nlocal = (*pWlocal).Iend - (*pWlocal).Istart;
    (*pWlocal).Nlocal = Nlocal;
    (*pWlocal).arr = malloc(sizeof(int*)*Nlocal);
    if ((*pWlocal).isBooleanMat == 0)
        (*pWlocal).arrv = malloc(sizeof(float*)*Nlocal);
    (*pWlocal).narr = malloc(sizeof(int)*Nlocal);
    (*pWlocal).nnz = malloc(sizeof(int)*Nlocal);
    for (j = 0; j<Nlocal; j++)
    {
        (*pWlocal).arr[j] = malloc(sizeof(int)*Wnnz[j]);
        if ((*pWlocal).isBooleanMat == 0)
            (*pWlocal).arrv[j] = malloc(sizeof(float)*Wnnz[j]);
        (*pWlocal).narr[j] = Wnnz[j];
        (*pWlocal).nnz[j] = 0;


    }

    return 1;
}



/*  pW is the local part of a global matrix partitioned by processors;
 idx1 is the global index (dimension 1)
 idx2 is the global index

 When setting values, it does whether there exist duplicative values already. 
 */
int MatSetValueT(struct Mat *pW, int idx1, int idx2, float value)
{
    int idx1local = idx1 - (*pW).Istart, rank;

    

    if ((*pW).nnz[idx1local] >= (*pW).narr[idx1local])
    {
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        printf("[%d] Number of none zeros (%d) at row %d (global) %d (local) exceeds the memory allocated (%d).  %d \n",  rank, (*pW).nnz[idx1local], idx1, idx1local, (*pW).narr[idx1local], (*pW).Istart);
        return -1;
    }
    if (idx1 >= (*pW).size[0])
    {
        printf("Index1 (global value %d, local value %d) exceeds the matrix size1 (%d). \n", idx1, idx1 - (*pW).Istart, (*pW).size[0]);
        return -1;
    }
    if (idx2 >= (*pW).size[1])
    {
        printf("Index2 (value %d at row %d (global) %d (local)) exceeds the matrix size2 (%d). \n", idx2, idx1, idx1 - (*pW).Istart, (*pW).size[1]);
        return -1;
    }

    if (value == 0)
        return 0;

    if ((*pW).isBooleanMat == 1)
        if (value != 1)
            printf("MatSetValue considers all the non-zero values to be true (1). \n");

    (*pW).arr[idx1local][(*pW).nnz[idx1local]] = idx2;
    if ((*pW).isBooleanMat == 0) /* otherwise, value has no effect */
        (*pW).arrv[idx1local][(*pW).nnz[idx1local]] = value;
    ((*pW).nnz[idx1local]) ++;

    return 0;



}


int  MatAssemble(struct Mat *pW) /* remove duplicative values in each row */
{

    int i, j, ncols, ncolsNew, sizeTMP;
    int *arrt, nnzMax;
    float *arrvt;
    struct PreLocSpace locT;


    if ((*pW).isBooleanMat == 0)
    {
        nnzMax = -1;
        for (i = 0; i<(*pW).Nlocal; i++)
            if (nnzMax < (*pW).nnz[i])
                nnzMax = (*pW).nnz[i];



        FnUniqueStart(&locT, nnzMax);
    }



    /* may remove the zero entries here */
    for (i = 0; i<(*pW).Nlocal; i++)
    {
        ncols = (*pW).nnz[i];

        if ((*pW).isBooleanMat == 0)
        {

            ncols = 0;/* to remove zeros */
            for (j = 0; j < (*pW).nnz[i]; j++)
            {
                if ((*pW).arrv[i][j] != 0)
                {
                    (*pW).arr[i][ncols] = (*pW).arr[i][j];
                    (*pW).arrv[i][ncols] = (*pW).arrv[i][j];
                    ncols ++;
                }
            }


            FnUnique((*pW).arr[i], ncols, &ncolsNew, (*pW).arrv[i], &locT);
        }

        if ((*pW).isBooleanMat == 1)
            FnUnique((*pW).arr[i], ncols, &ncolsNew, NULL, NULL);
        sizeTMP = sizeof(int)*ncolsNew;
        arrt = malloc(sizeTMP);
        memcpy(arrt, (*pW).arr[i], sizeTMP);



        free((*pW).arr[i]);

        (*pW).arr[i] = arrt;
        (*pW).narr[i] = ncolsNew;
        (*pW).nnz[i] = ncolsNew;

        if ((*pW).isBooleanMat == 0)
        {
            sizeTMP = sizeof(float)*ncolsNew;
            arrvt = malloc(sizeTMP);
            memcpy(arrvt, (*pW).arrv[i], sizeTMP);
            free((*pW).arrv[i]);
            (*pW).arrv[i] = arrvt;
        }


    }




    if ((*pW).isBooleanMat == 0)
    {
        FnUniqueTerm(&locT);

    }


    return 0;

}


int MatDestroyT(struct Mat *pW)
{
    int i;

    for (i = 0; i<(*pW).Nlocal; i++)
    {
        free((*pW).arr[i]);
        if ((*pW).isBooleanMat == 0)
            free((*pW).arrv[i]);
    }
    free((*pW).arr);
    if ((*pW).isBooleanMat == 0)
        free((*pW).arrv);
    free((*pW).narr);
    free((*pW).nnz);

    return 0;


}

int MatViewGlobal(struct Mat *pWlocal) /* view the global matrix */
{
    int size, rank, i, iLocal, j;
    MPI_Comm *pcomm = &((*pWlocal).comm);

    MPI_Comm_size(*pcomm,&size);
    MPI_Comm_rank(*pcomm,&rank);





    for (i = 0; i<size; i++)
    {
        if (rank == i)
        {
            for (iLocal = 0; iLocal < (*pWlocal).Nlocal; iLocal++)
            {

                printf("Line %d: ", iLocal + (*pWlocal).Istart);
                for (j = 0; j<(*pWlocal).nnz[iLocal]; j++) /*  before assembling, max(j)+1 is equal to nnz; after assembling, it's equal to nnz and narr) */
                {
                    printf("[%d] ", (*pWlocal).arr[iLocal][j]);

                    if ((*pWlocal).isBooleanMat == 0)
                        printf("%1.2f ", (*pWlocal).arrv[iLocal][j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(*pcomm);
    }


    return 0;


}




int FnComputeComponent(struct Mat* adj, int i, int* plabelall, int labeli, struct Queue *q) /* for node i in graph adj, set all the cluster labels to labeli */
{
    int newnode, newnodeB, isSuccess, j, re = 0;

    newnodeB = i;
    FnQueuePush(q, newnodeB);

    while(1)
    {

        FnQueuePop(q, &newnode, &isSuccess);
        if (isSuccess == 0)
            break;



        plabelall[newnode] = labeli;
        for (j = 0; j < (*adj).narr[newnode]; j++)
        {
            newnodeB = (*adj).arr[newnode][j];
            if (plabelall[newnodeB] != labeli)
                re = FnQueuePush(q, newnodeB);
        }


        if (re == -1)
            return -1;


    }

    return 0;
}


int FnQueueCreate(struct Queue *q, int size)
{


    (*q).size = size;
    (*q).sizealloc = size + 1;
    (*q).arr = malloc(sizeof(int)*(*q).sizealloc);
    (*q).qstart = 0;
    (*q).qend = 0;

    return 0;
}


int FnQueuePush(struct Queue *q, int value)
{
    int *pstart = &((*q).qstart), *pend = &((*q).qend) /* qend is the exact ending location plus 1 */, newend;

    newend = *pend + 1;
    if (newend > (*q).size)/* starting from zero to size, so the total number of entries is sizealloc */
        newend = 0;

    if (newend == *pstart)
    {
        printf("Queue is full: size = %d, size allocated = %d, start = %d, end = %d.\n ", (*q).size, (*q).sizealloc, (*q).qstart, (*q).qend);
        return -1;
    }

    (*q).arr[*pend] = value;
    *pend = newend;

    return 0;


}


int FnQueuePop(struct Queue *q, int* value, int* isSuccess)
{
    int *pstart = &((*q).qstart), *pend = &((*q).qend) /* qend is the exact ending location plus 1 */;

    if (*pstart == *pend)
    {
        /* queue is empty */
        *isSuccess = 0;
        return 0;
    }

    *isSuccess = 1;
    *value = (*q).arr[*pstart];
    *pstart = *pstart + 1;
    if (*pstart > (*q).size)/* starting from zero to size, so the total number of entries is sizealloc */
        *pstart = 0;

    return 0;


}

int FnQueueView(struct Queue *q)
{
    int i;

    i = (*q).qstart;
    while(1)
    {
        if (i == (*q).qend) /* exceeds the exact ending location */
            break;


        printf("%d ", (*q).arr[i]);
        i++;
        if (i>(*q).size)/* can be equal because the actual queue size is one larger than q.size */
            i = 0;

    }

    printf("\n");

    return 0;
}


int FnQueueDestroy(struct Queue *q)
{
    free((*q).arr);
    (*q).size = 0;
    (*q).sizealloc = 0;
    (*q).qstart = -1;
    (*q).qend = -1;


    return 0;
}




int sortint_index(int ncols, int* vec, int* orderIdx, int isChangeVec /* 1, change vec to be sorted; 0, don't change */)
{
    int i;
    struct SortEleI *pe;

    pe = malloc(sizeof(struct SortEleI)*ncols);

    for (i = 0; i<ncols; i++)
    {
        pe[i].value = vec[i];
        pe[i].index = orderIdx[i];
    }

    qsort(pe, ncols, sizeof(struct SortEleI), FnCmpFunInt_index);

    for (i = 0; i<ncols; i++)
    {
        if (isChangeVec == 1)
            vec[i] = pe[i].value;
        orderIdx[i] = pe[i].index;
    }

    free(pe);
    return 0;

}


int FnCmpFunInt_index (const void * a, const void * b)
{
    return ( (*(struct SortEleI*)a).value - (*(struct SortEleI*)b).value );
}


int sortfloat_index(int ncols, float* vec, int* orderIdx, int isChangeVec /* 1, change vec to be sorted; 0, don't change */)
{
    int i;
    struct SortEleF *pe;

    pe = malloc(sizeof(struct SortEleF)*ncols);

    for (i = 0; i<ncols; i++)
    {
        pe[i].value = vec[i];
        pe[i].index = orderIdx[i];
    }

    qsort(pe, ncols, sizeof(struct SortEleF), FnCmpFunfloat_index);

    for (i = 0; i<ncols; i++)
    {
        if (isChangeVec == 1)
            vec[i] = pe[i].value;
        orderIdx[i] = pe[i].index;
    }


    free(pe);
    return 0;

}


int FnCmpFunfloat_index (const void * a, const void * b)
{
    float dt =  (*(struct SortEleF*)a).value - (*(struct SortEleF*)b).value;
    /* the return value is int, so don't return dt directly */


    if (dt > 0)
        return 1;

    if (dt < 0)
        return -1;


    return 0;


}



