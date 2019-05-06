#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sstream>
#include <unistd.h>

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
    
    if(tId<M*N){
        par=int(tId/N)*N+((2*int(tId/X))*X+tId%X)%N;
        impar=int(tId/N)*N+(((2*int(tId/X)+1))*X+tId%X)%N;
        
        if((blockIdx.x)%4 < 2){
            Rout[impar]=R[par]; 
            Gout[impar]=G[par];
            Bout[impar]=B[par];
        }
        else{
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

    for(int i = 0; i < imsize; i++){
        if(i%(*N)==0){
            prim=0;
            sec=(*N)/2;
        }
        if ((i%(2*X))<X){
            
            
            fscanf(fp, "%f ", &(R1[prim+((i/(*N))*(*N))]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(R1[sec+((i/(*N))*(*N))]));
            sec++;
        }

    }

    for(int i = 0; i < imsize; i++){
        if(i%(*N)==0){
            prim=0;
            sec=(*N)/2;
        }
        if ((i%(2*X))<X){
            fscanf(fp, "%f ", &(G1[prim+((i/(*N))*(*N))]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(G1[sec+((i/(*N))*(*N))]));
            sec++;
        }       
    }

    for(int i = 0; i < imsize; i++){
        if(i%(*N)==0){
            prim=0;
            sec=(*N)/2;
        }
        if((i%(2*X))==0 && (i/(*N))<4){
                std::cout<<(i/(*N))<<":"<<prim+((i/(*N))*(*N))<<"\n";
                               
            }
        
        if ((i%(2*X))<X){
            fscanf(fp, "%f ", &(B1[prim+((i/(*N))*(*N))]));
            prim++;
        }
        else{
            fscanf(fp, "%f ", &(B1[sec+((i/(*N))*(*N))]));
            sec++;
        }       
    }
    
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
    
}

__global__ void kernel3(float *R, float *G, float* B, float *Rout, float *Gout, float* Bout, int M, int N, int X){
    int tId= threadIdx.x+blockIdx.x*blockDim.x;
    int par, impar;

    if(tId<M*N){
        par=int(tId/N)*N+((2*int(tId/X))*X+tId%X)%N;
        impar=int(tId/N)*N+(((2*int(tId/X)+1))*X+tId%X)%N;
        
        if((blockIdx.x)%4 < 2){
            Rout[impar]=R[tId]; 
            Gout[impar]=G[tId];
            Bout[impar]=B[tId];
        }
        else{          
            Rout[par]=R[tId]; 
            Gout[par]=G[tId];
            Bout[par]=B[tId];
        }
    }
    
}


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
    Read(&Rhost, &Ghost, &Bhost, &M, &N, "imagen.txt");
    

    /*
     *  Parte CPU
    */
    std::cout <<"Pregunta 1" << std::endl;

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
        ss << "imgCPU-P1-X_" << X << ".txt";
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

    std::cout <<"Pregunta 2" << std::endl;

    //  Primer Kernel 
    for( X=1; X<1024; X*=2){
        ss.str("");
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernel1<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, X);
        cudaDeviceSynchronize();
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
        std::cout <<"X:"<< X<< "-Tiempo GPU: " << dt << "[ms]" << std::endl;
        ss << "imgGPU-P2-X_" << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
    
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    
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

    std::cout <<"Pregunta 3" << std::endl;
    
    /*Segundo Kernel*/

    for( X=1; X<1024; X*=2){
        ss.str("");
        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);
        cudaEventRecord(ct1);
        kernel2<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout,Gdevout,Bdevout, M, N, X);
        cudaEventRecord(ct2);
        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
        cudaDeviceSynchronize();
        std::cout <<"X:"<< X<< "-Tiempo GPU: " << dt << "[ms]" << std::endl;
        ss << "imgGPU-P3-X_" << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }
        
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    
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


    std::cout <<"Pregunta 4" << std::endl;

    //Tercer Kernel
    
    for( X=1; X<1024; X*=2){
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
        std::cout <<"X:"<< X<< "-Tiempo GPU: " << dt << "[ms]" << std::endl;
        ss << "imgGPU-P4-X_" << X << ".txt";
        s = ss.str();
        cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        Write(Rhostout, Ghostout, Bhostout, M, N, s.c_str());
    }

    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    


    return 0;
}