#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sstream>

void Read(float** R, float** G, float** B, int *M, int *N, const char *filename) {    
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

void Read_scanf(float** R, float** G, float** B, int *M, int *N) {    

    scanf("%d %d\n", M, N); // read firts three importand values

    int imsize=(*M)*(*N);

    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
    
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(R1[i]));
    }
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(G1[i]));
    }
    for(int i=0; i < imsize; ++i){ // iteration for the line with m*n float values // COLOR R
        scanf("%f ", &(B1[i]));                         
    }
    
    *R = R1; *G = G1; *B = B1;
}

/*
 *  Escritura Archivo
 */

void Write(float* R, float* G, float* B, int M, int N, const char *filename) {
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

/*
 *  Procesamiento Imagen CPU
 */
void funcionCPU(float *R,float *G, float* B,int M,int N, int X,float *Rout,float* Gout,float* Bout){
    for(int i=0;i<M*N;++i){
        if((i%(2*X))<X){
            Rout[i]=R[i+X]; 
            Gout[i]=G[i+X];
            Bout[i]=B[i+X];
        }
        else{
            Rout[i]=R[i-X]; 
            Gout[i]=G[i-X];
            Bout[i]=B[i-X];
        }
    } 
}


/*
 *  Procesamiento Imagen GPU
 */

/*Pregunta 2*/
__global__ void kernel1(float *R, float *G, float* B, float *Rout, float *Gout, float* Bout, int M, int N, int X){
    
    int tId= threadIdx.x+blockIdx.x*blockDim.x;
    
    if(tId<M*N){
        if((tId%(2*X))<X){
            Rout[tId]=R[tId+X]; 
            Gout[tId]=G[tId+X];
            Bout[tId]=B[tId+X];
        }
        else{
            Rout[tId]=R[tId-X]; 
            Gout[tId]=G[tId-X];
            Bout[tId]=B[tId-X];
        }
    }
    
    
}
/*Pregunta 3*/
__global__ void kernel2(float *R, float *G, float* B, float *Rout, float *Gout, float* Bout, int M, int N, int X){
    int tId= threadIdx.x+blockIdx.x*blockDim.x;
    int par, impar;
    int shift=(M*N)/2;
    
    
    if(tId<M*N){
        
        if(blockIdx.x < 2){
            par=2*(tId/X)*X+tId%X;
            impar=(2*(tId/X)+1)*X+tId%X;          
            
            Rout[impar]=R[par]; 
            Gout[impar]=G[par];
            Bout[impar]=B[par];
        }
        else{
            par=(2*(tId/X)-shift)*X+tId%X;
            impar=((2*(tId/X)+1)-shift)*X+tId%X;           
            //printf("bloque:%d accede:%d escribe:%d\n",blockIdx.x, impar%N, par%N);
            Rout[par]=R[impar]; 
            Gout[par]=G[impar];
            Bout[par]=B[impar];
        }
    }
   
}

void Read2(float** R, float** G, float** B, int *M, int *N,int X, const char *filename){
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", M, N);
    int prim, sec;

    int imsize = (*M) * (*N);
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
    prim=0;
    sec=(*N)/2;
    for(int i = 0; i < imsize; i++){
        if ((i%(2*X))<X){
            fscanf(fp, "%f ", &(R1[prim]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(R1[sec]));
            sec++;
        }       
    }
    prim=0;
    sec=(*N)/2;
    for(int i = 0; i < imsize; i++){
        if ((i%(2*X))<X){
            fscanf(fp, "%f ", &(G1[prim]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(G1[sec]));
            sec++;
        }       
    }
    prim=0;
    sec=(*N)/2;
    for(int i = 0; i < imsize; i++){
        if ((i%(2*X))<X){
            fscanf(fp, "%f ", &(B1[prim]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(B1[sec]));
            sec++;
        }       
    }
    
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
    
}


/*
__global__ void kernel3(float *R, float *G, float* B, float *Rout, float *Gout, float* Bout, int M, int N, int X){
    int tId= threadIdx.x+blockIdx.x*blockDim.x;
    int par, impar;
    if(tId<M*N){
        
        if(blockIdx.x < 2){
            par=2*(tId/X)*X+tId%X;
            impar=(2*(tId/X)+1)*X+tId%X;
            Rout[impar]=R[tId]; 
            Gout[impar]=G[tId];
            Bout[impar]=B[tId];
        }
        else{
            
            par=(2*(tId/X)-shift)*X+tId%X;
            impar=((2*(tId/X)+1)-shift)*X+tId%X;            
            Rout[par%N]=R[tId]; 
            Gout[par%N]=G[tId];
            Bout[par%N]=B[tId];
        }
    }
    
}
*/


/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
    clock_t t1, t2;
    cudaEvent_t ct1, ct2;
    double ms;
    float dt;
    int M, N, X;
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    float *Rdev, *Gdev, *Bdev;
    float *Rdevout, *Gdevout, *Bdevout;

    
    // Lectura de datos
    //Read(&Rhost, &Ghost, &Bhost, &M, &N, "imagen.txt");
    Read_scanf(&Rhost, &Ghost, &Bhost, &M, &N);
    

    /*
     *  Parte CPU
    */
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N]; 
    
    std::stringstream ss;    
    std::string s;
    
    
    for(int X=1; X<1024; X*=2){
        ss.str("");
        t1 = clock(); 
        funcionCPU(Rhost, Ghost, Bhost, M, N, X,Rhostout, Ghostout, Bhostout); // Agregar parametros!
        t2 = clock();
        ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
        std::cout <<"X:"<< X<< "-Tiempo CPU: " << ms << "[ms]" << std::endl;
        ss << "imgCPU-X_ " << X << ".txt";
        s = ss.str();
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
    
    

    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    
    /*
     *  Parte GPU
     */    
    int grid_size, block_size = 256;
    grid_size = (int)ceil((float) M * N / block_size);
        
    cudaMalloc((void**)&Rdev, M * N * sizeof(float));
    cudaMalloc((void**)&Gdev, M * N * sizeof(float));
    cudaMalloc((void**)&Bdev, M * N * sizeof(float));
    cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Gdev, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdev, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
    cudaMalloc((void**)&Rdevout, M * N * sizeof(float));
    cudaMalloc((void**)&Gdevout, M * N * sizeof(float));
    cudaMalloc((void**)&Bdevout, M * N * sizeof(float));
    
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];

    /* Primer Kernel */
    for( X=1; X<1024; X*=2){
        ss.str("");
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernel1<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, X); // Agregar parametros!
        cudaDeviceSynchronize();
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
        std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;
        ss << "imgGPU1-X_ " << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
    
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];

    /*
    
    /*Segundo Kernel*/
    /*for( X=1; X<1024; X*=2){
        ss.str("");
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernel2<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, X); // Agregar parametros!
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
        cudaDeviceSynchronize();
        std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;
        ss << "imgGPU2-X_ " << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
    
    for (int i = 1000; i < 1100; ++i)
    {
        std::cout<< '-'<<Rhostout[i];
    }
    
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];*/

    
    /*Tercer Kernel*/
    
    /*for( X=1; X<1024; X*=2){
        ss.str("");
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        Read2(&Rhost, &Ghost, &Bhost, &M, &N, X, "imagen.txt");

        cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Gdev, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bdev, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);

        kernel3<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, X); // Agregar parametros!
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
        cudaDeviceSynchronize();
        std::cout << "Tiempo GPU3: " << dt << "[ms]" << std::endl;
        ss << "imgGPU3-X_ " << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
    */

    
    ss.str("");
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernel2<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, 2); // Agregar parametros!
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaDeviceSynchronize();
    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;
    ss << "imgGPU222-X_ " << 64 << ".txt";
    s = ss.str();
    cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());

    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    delete[] Rhost; delete[] Ghost; delete[] Bhost;
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;



    return 0;
}