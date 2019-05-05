
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>


void Write(float* R, float* G, float* B, 
	       int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
}
void Read(float** R, float** G, float** B, 
          int *M, int *N, const char *filename) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", M, N);

    int imsize = (*M) * (*N);
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(R1[i]));
    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(G1[i]));
    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(B1[i]));
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
}

void Read_scanf(float** R, float** G, float** B, 
          int *M, int *N) {    

    scanf("%d %d\n", M, N); // read firts three importand values
 
    int arrsize =(*M)*(*N);
    int imsize=(*M)*(*N);

    float* R1 = new float[arrsize];
    float* G1 = new float[arrsize];
    float* B1 = new float[arrsize];
    
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(R1[i+imsize]));
    }
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(G1[i+imsize]));
    }
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(B1[i+imsize]));                         
    }
    
    *R = R1; *G = G1; *B = B1;
}

__global__ void kernelGPU(float *R,float* G,float* B,float* Rin,float*Gin,float*Bin,int M,int N,int L){

    int tId= threadIdx.x+blockIdx.x*blockDim.x;
    int i;    
    if(tId<M*N){
        R[tId]=0;
        G[tId]=0;
        B[tId]=0;
        for(i=0; i<L; ++i ){
            
            R[tId]+= Rin[tId+i*M*N];
            G[tId]+= Gin[tId+i*M*N];
            B[tId]+= Bin[tId+i*M*N];
        }
        
        R[tId]=R[tId]/L;
        G[tId]=G[tId]/L;
        B[tId]=B[tId]/L;
       }
        
        
    }

void funcionCPU(float *R,float* G,float* B,float* Rin,float*Gin,float*Bin,int M,int N,int L){
    
    int i,j;
    for (i=0;i<M*N;++i){
        for(j=0; j<L; ++j){
            R[i]+= Rin[i+j*M*N];
            G[i]+= Gin[i+j*M*N];
            B[i]+= Bin[i+j*M*N];
        }
        R[i]=R[i]/L;
        G[i]=G[i]/L;
        B[i]=B[i]/L;
    }
    
        
}

int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	clock_t t1, t2;
	cudaEvent_t ct1, ct2;
	double ms;
	float dt;
	int M, N, L;
    float *Rhost, *Ghost, *Bhost;  
    float *Rhostout, *Ghostout, *Bhostout;
    
    
    float *R, *G, *B; //Resultado
    float *Rin, *Gin, *Bin; //Entrada

    Read_scanf(&Rhost, &Ghost, &Bhost, &M, &N, &L);


    /*
     *  Parte CPU
     */
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];
    

    t1 = clock();
    funcionCPU(Rhostout, Ghostout, Bhostout,Rhost, Ghost, Bhost, M, N, L); // Agregar parametros!
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    std::cout << "Tiempo CPU: " << ms << "[ms]" << std::endl;
    Write(Rhostout, Ghostout, Bhostout, M, N, "imgCPU.txt");

    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;

    /*  Parte GPU
     */

    int grid_size, block_size = 256;
    grid_size = (int)ceil((float) M * N/ block_size);
        
    cudaMalloc((void**)&Rin, M * N * sizeof(float)*L);
    cudaMalloc((void**)&Gin, M * N * sizeof(float)*L);
    cudaMalloc((void**)&Bin, M * N * sizeof(float)*L);
    cudaMemcpy(Rin, Rhost, M * N * sizeof(float)*L, cudaMemcpyHostToDevice);
    cudaMemcpy(Gin, Ghost, M * N * sizeof(float)*L, cudaMemcpyHostToDevice);
    cudaMemcpy(Bin, Bhost, M * N * sizeof(float)*L, cudaMemcpyHostToDevice);
        
    cudaMalloc((void**)&R, M * N * sizeof(float));
    cudaMalloc((void**)&G, M * N * sizeof(float));
    cudaMalloc((void**)&B, M * N * sizeof(float));
    
    
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernelGPU<<<grid_size, block_size>>>(R, G, B, Rin,Gin,Bin, M, N, L); // Agregar parametros!
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;
    
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];
    cudaMemcpy(Rhostout, R, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ghostout, G, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhostout, B, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    Write(Rhostout, Ghostout, Bhostout, M, N, "imgGPU.txt");
    
    cudaFree(R); cudaFree(G); cudaFree(B);
    cudaFree(Rin); cudaFree(Gin); cudaFree(Bin);
    delete[] Rhost; delete[] Ghost; delete[] Bhost;
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	return 0;
} 
