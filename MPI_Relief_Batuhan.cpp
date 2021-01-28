#include <iostream>
#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>


using namespace std;

double manhattan(double **twodarray, int current, int other , int A);
int findNearestHit(double** twodarray, int cinstance, int eachProc, int A);
int findNearestMiss(double** twodarray, int cinstance, int eachProc, int A);
void maxandminfinder(double **twodarray, int eachProc, int A, double maxes[], double mins[]);
void sort(int results[], int T);
int uniquevals(int size, int array[], int temp[]);

/*
 * Author: Batuhan Tongarlak
 * January 2021
 */
int main(int argc, char *argv[]) {

    int P, N, A, M, T; // Variables
    int eachProc;      // # of instances each process receive

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0){          // part of the Master process' tasks


        ifstream file(argv[1]);      // Attaining the values of variables
        int inits;
        file >> inits;
        P = inits;
        file >> inits;
        N = inits;
        file >> inits;
        A = inits;
        file >> inits;
        M = inits;
        file >> inits;
        T = inits;

        eachProc = N / (P-1);       // Attaining the the number of process' each slave'S going to take
        double element;             //
        double **instances;                     // dynamic 2d array for instances
        instances = new double *[N];            //
        for(int i = 0; i <N; i++){              //
            instances[i] = new double[A+1];}    //

        for (int i = 0; i < N; ++i) {           // mapping NxA+1 instance input into the 2d array(instances[][])
            for (int j = 0; j < A+1; ++j) {     //
                file >> element;                //
                instances[i][j]=element;        //
            }                                   //
        }                                       //

        MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);           // Broadcasting the data of
        MPI_Bcast(&eachProc, 1, MPI_INT, 0, MPI_COMM_WORLD);    // A, M, T, N and eachProc to the slaves
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);           //
        MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);           //
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);           //

        for (int i = 0; i < P-1; ++i) {                                    // Sending all the values at the NxA+1 input one by one
            for (int j = i*eachProc; j < (i+1)*eachProc; ++j) {            // to the corresponding slaves regarding to eachProc value.
                for (int k = 0; k < A+1; ++k) {                            //
                    MPI_Send(                                              //
                            /* data         = */ &instances[j][k],         //
                            /* count        = */ 1,                        //
                            /* datatype     = */ MPI_DOUBLE,               //
                            /* destination  = */ i+1,                 //
                            /* tag          = */ 0,                        //
                            /* communicator = */ MPI_COMM_WORLD);          //
                }
            }
        }

        int result;   // an object that used for receiving result data from slaves
        int *masterresults = new int[(P-1)*T];     // dynamic array for master's output

        for (int i = 0; i < P-1; ++i) {                                     //      Receiving (by Master) the results of the slave process' one by one
            for (int j = 0; j < T; ++j) {                                   //
                MPI_Recv(                                                   //
                        /* data         = */ &result,                       //
                        /* count        = */ 1,                             //
                        /* datatype     = */ MPI_INT,                       //
                        /* source       = */ i+1,                           //
                        /* tag          = */ 0,                             //
                        /* communicator = */ MPI_COMM_WORLD,                //
                        /* status       = */ MPI_STATUS_IGNORE);            //
                masterresults[i*(T)+j]=result;                              //
            }                                                               //
        }                                                                   //
        int *temp = new int[T*(P-1)];       // a temporary array used for finding the unique values of the results came from slaves
        int size = uniquevals(T*(P-1), masterresults,temp);  // size is the number of the unique values, it's explained more detailed at uniquevals function

        int *lastmaster = new int[size];        // dynamic array used for arranging the master's output

        for (int i = 0; i < size; ++i) {        // arranging lastmaster array regarding to the size of the unique values
            lastmaster[i]=temp[i];              //
        }                                       //

        sort(lastmaster,size);                  // sorting the master's output

        cout << "Master P0 : ";                 // print operations
        for (int i = 0; i < size; ++i) {        //
            cout << lastmaster[i] << " " ;      //
        }                                       //
        cout << endl;                           //
    }                                           //

    else{       // part of the slaves (because their ranks are larger than 0)

        MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);           // Slaves can operate with A, M, T, N and eachProc
        MPI_Bcast(&eachProc, 1, MPI_INT, 0, MPI_COMM_WORLD);    // thanks to these Broadcast functions.
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);           //
        MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);           //
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);           //

        double number;          // a double object used for receiving data
        double **sepInstances;                      // 2d double array for the slaves and their corresponding inputs.
        sepInstances = new double *[eachProc];      // sepInstances -> Seperated Instances
        for(int i = 0; i <eachProc; i++){           //
            sepInstances[i] = new double[A+1];}     //

        double maxes[A], mins[A];                   // max and min arrays for each feature to save their max and min values

        for (int i = 0; i < eachProc; ++i) {                    //  Slaves are receiving the corresponding input parts here one by one
            for (int j = 0; j < A+1; ++j) {                     //  The datum has newly received will be mapped to the corresponding location
                MPI_Recv(                                       //  of the 2d array(sepInstances[][]).
                        /* data         = */ &number,           //
                        /* count        = */ 1,                 //
                        /* datatype     = */ MPI_DOUBLE,        //
                        /* source       = */ 0,                 //
                        /* tag          = */ 0,                 //
                        /* communicator = */ MPI_COMM_WORLD,    //
                        /* status       = */ MPI_STATUS_IGNORE);//
                sepInstances[i][j]=number;                      //
            }
        }
        maxandminfinder(sepInstances,eachProc,A,maxes,mins);    // this function finds the max and mins of the features and fills the arrays.

        double W[A];                        // Weight array, size is A (number of features)
        for (int i = 0; i < A; ++i) {       // Every indexes value is initialized with 0
            W[i]=0;                         //
        }                                   //
        for (int i = 0; i < M; ++i) {                                               // beginning of the relief algorithm, with M iterations

            int hitInstance=findNearestHit(sepInstances, i, eachProc, A);        // it finds the nearest hit instance and assigns it to the hitInstance value
            int missInstance=findNearestMiss(sepInstances,i, eachProc, A);       // it finds the nearest miss instance and assigns it to the hitInstance value

            for (int a = 0; a < A; ++a) {                                                                              // this part calculates and updates the weights
                W[a] = W[a] - abs(sepInstances[i][a]-sepInstances[hitInstance][a])/(M*(maxes[a]-mins[a])) +         // of the features
                       abs(sepInstances[i][a]-sepInstances[missInstance][a])/(M*(maxes[a]-mins[a]));                //
            }                                                                                                          //
        }

        int *results= new int[T];               // dynamic results array for the top T features for each slave.

        double max=-1;                          // part where the top T weights are found and assigned to the results array
        for (int i = 0; i < T; ++i) {           //
            for (int j = 0; j < A; ++j) {       //
                if(max<W[j]){                   //
                    max=W[j];                   //
                    results[i] = j;             //
                }                               //
            }                                   //
            max = -1;                           //
            W[results[i]] = -1;                 //
        }

        sort(results,T);                        // sorting of the results array

        cout << "Slave P" << rank << " : ";     // printing operations for slaves
        for (int i = 0; i < T; ++i) {           //
            cout << results[i] << " ";          //
        }                                       //
        cout << endl;                           //


        for (int i = 0; i < T; ++i) {                           // Sending the result arrays' elements one by one to the master
            MPI_Send(                                           //
                    /* data         = */ &results[i],           //
                    /* count        = */ 1,                     //
                    /* datatype     = */ MPI_INT,               //
                    /* destination  = */ 0,                //
                    /* tag          = */ 0,                     //
                    /* communicator = */ MPI_COMM_WORLD);       //
        }                                                       //
    }
    MPI_Barrier(MPI_COMM_WORLD);                // used for synchronization
    MPI_Finalize();                             // Finalization of the MPI operation
    return 0;
}
/**
 * This function finds the nearest hit instance for given target instance and returns its index
 * @param twodarray the 2d array of slaves seperated inputs
 * @param cinstance target instances index
 * @param eachProc number of instances each slave has to handle
 * @param A number of features
 * @return nearest hits index
 */
int findNearestHit(double **twodarray , int cinstance, int eachProc, int A){

    double nearestHit=INFINITY;
    int nHitInstance=0;
    for (int j = 0; j < eachProc; ++j) {
        if(cinstance!=j){
            if(twodarray[cinstance][A]==twodarray[j][A]){
                double distance = manhattan(twodarray, cinstance, j, A);
                if (nearestHit > distance) {
                    nearestHit = distance;
                    nHitInstance = j;
                }
            }
        }
    }
    return nHitInstance;
}
/**
 * This function finds the nearest miss instance for given target instance and returns its index
 * @param twodarray the 2d array of slaves seperated inputs
 * @param cinstance target instances index
 * @param eachProc number of instances each slave has to handle
 * @param A number of features
 * @return nearest miss index
 */
int findNearestMiss(double **twodarray , int cinstance, int eachProc, int A) {

    double nearestMiss = INFINITY;
    int nMissInstance = 0;
    for (int i = 0; i < eachProc; ++i) {
        if (cinstance != i) {
            if (twodarray[cinstance][A] != twodarray[i][A]) {
                double distance = manhattan(twodarray, cinstance, i, A);
                if (nearestMiss > distance) {
                    nearestMiss = distance;
                    nMissInstance = i;
                }
            }
        }
    }
    return nMissInstance;
}
/**
 * it calculates the manhattan distance between two instances (target and the other)
 * @param twodarray the 2d array of slaves seperated inputs
 * @param current target instance
 * @param other other instance
 * @param A number of the features
 * @return
 */
double manhattan(double **twodarray, int current, int other , int A){
    double sum=0;
    for (int i = 0; i < A; ++i) {
        sum=sum+fabs(twodarray[current][i]-twodarray[other][i]);
    }
    return sum;
}
/**
 * it finds the max and min values for all the NxA+1 inputs and fills the given maxes and mins arrays
 * @param twodarray the 2d array of slaves seperated inputs
 * @param eachProc number of instances each slave has to handle
 * @param A number of the features
 * @param maxes empty max array
 * @param mins empty min array
 */
void maxandminfinder(double **twodarray, int eachProc, int A, double maxes[], double mins[]){

    for (int i = 0; i < A; ++i) {
        maxes[i]=twodarray[0][i];
        mins[i]=twodarray[0][i];
    }

    for (int i = 0; i < A; ++i) {
        for (int j = 0; j < eachProc; ++j) {
            double cavy = twodarray[j][i];
            if(maxes[i]<cavy){
                maxes[i]=cavy;
            }
            if(mins[i]>cavy){
                mins[i]=cavy;
            }
        }
    }
}
/**
 * sorting function
 * @param results empty results array
 * @param T number of the features desired
 */
void sort(int results[], int T){

    int temp;
    for(int i=0; i<T; i++)
    {
        for(int j=i+1; j<T; j++)
        {
            //If there is a smaller element found on right of the array then swap it.
            if(results[j] < results[i])
            {
                temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
}
/**
 * function that finds the unique values and wrotes them to the given temp array
 * @param size size of the array
 * @param array given array with possible replications
 * @param temp empty temp array (after this function it will be fulfilled)
 * @return it returns the number of the unique values
 */
int uniquevals(int size, int array[], int temp[]) {

    int k = 0;
    for (int i = 0; i < size; i++) {
        int j;
        for (j = 0; j < i; j++)
            if (array[i] == array[j])
                break;
        if (i == j) {
            temp[k] = array[i];
            k++;
        }
    }

    return k;
}
