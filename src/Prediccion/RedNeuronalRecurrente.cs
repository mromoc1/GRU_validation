using System;
using System.IO;
namespace DISTO_DMH_SW2{
    class RedNeuronalRecurrente{
        private const string ruta_pesosred = @"./src/Data/pesosred.txt";
        private Logica logica = new Logica();
        private FuncionesActivacion funcionesActivacion = new FuncionesActivacion();
        //Variables a utilizar
        private float [,] Wz;
        private float [,] Wr;
        private float [,] Wc;
        private float [,] Uz;
        private float [,] Ur;
        private float [,] Uc;
        private float [,] V;
        private float[] biasZ;
        private float[] biasR;
        private float[] biasC;
        private float[] biasV;
        private float[] C; // ct // content
        private float[] X; // entrada
        private float[] Z; // z(t) puerta //update
        private float[] R; // r(t) puerta //reset
        private float[] H; // s(t) // ht // hidden
        private int longitud_Xt = 0;
        private int longitud_Diccionario = 0;
        public RedNeuronalRecurrente(Boolean cargarRNN,int Ocultas,int Salidas){
            longitud_Xt = Ocultas;
            longitud_Diccionario = Salidas;
            if(cargarRNN){
                cargarRed();
            }else{
                inicializarRed();
            }
        }
        private void inicializarRed(){
            //matrices
            Wz = generarMatriz("Compuertas");
            Wr = generarMatriz("Compuertas");
            Wc = generarMatriz("Compuertas");
            Uz = generarMatriz("Compuertas");
            Ur = generarMatriz("Compuertas");
            Uc = generarMatriz("Compuertas");
            V = generarMatriz("Compuertas"); //**7&%^$%^@#$%$
            //bias 
            biasZ = generarVector("bias");
            biasR = generarVector("biasReset");
            biasC = generarVector("bias");
            biasV = generarVector("bias");
            //
            H = new float[biasZ.Length];//hidden
            X = new float[biasZ.Length];//vector de entrada a gru en el instante t
            Z = new float[biasZ.Length];//update gate
            R = new float[biasZ.Length];//reset gate
            C = generarVector("inicial"); // content
        }
        public float [] feedForward(float[] Xt){
            X = Xt;//vector de entrada de la red
            Z = updateResetGate(Uz, X, Wz, H, biasZ);
            R = updateResetGate(Ur, X, Wr, H, biasR);
            C = contentGate(Uc, Xt, Wc, H, R ,biasC);
            H = HiddenGate(Z, C, H);
            float[] inputSoftmax = logica.multiplcarMatrizVector(V,H);
            float[] O = funcionesActivacion.softmax(logica.sumaVectores(inputSoftmax,biasV));
            return O;
        }
        private float[] updateResetGate(float[,] U, float[] X, float[,] W,float[] H, float[] bias){
            float[] UporX = logica.multiplcarMatrizVector(U,X);
            float[] WporH = logica.multiplcarMatrizVector(W,H); 
            float[] sumaUXWc = logica.sumaVectores(UporX,WporH);
            float[] sumaBias = logica.sumaVectores(sumaUXWc, bias);
            float[] sigmoide = funcionesActivacion.Sigmoide(sumaBias);
            return sigmoide;
        }
        private float[] contentGate(float[,] Uc,float[] X, float[,] Wc, float[] H, float[] Rt, float[] bias){
            float[] UporX = logica.multiplcarMatrizVector(Uc,X);
            float[] prodHadSZt = logica.productoHadamard(H,Rt);
            float[] WcporSZt = logica.multiplcarMatrizVector(Wc,prodHadSZt);
            float[] suma = logica.sumaVectores(UporX, WcporSZt);
            float[] suma2 = logica.sumaVectores(suma,bias);
            float[] tanh = funcionesActivacion.tangenteHiperbolica(suma2);
            return tanh;
        }
        private float[] HiddenGate(float[] Zt, float[] C, float[] H){
            float[] resta = logica.resta(Zt);
            float[] ph1 = logica.productoHadamard(resta, C);
            float[] ph2 = logica.productoHadamard(Zt,H);
            float[] suma = logica.sumaVectores(ph1,ph2);
            return suma;
        }
        private float[] generarVector(string tipo){
            Random random = new Random();
            float[] vector = new float[longitud_Xt];
            float[] vector2 = new float[longitud_Diccionario];
            for (int i = 0; i < vector.Length; i++){
                switch (tipo){
                    case "inicial": 
                        vector[i] = 0; 
                    break;
                    case "noInicial": 
                        vector[i] = ((float)random.NextDouble()-0.4f)*0.1f;
                    break;
                    case "bias": 
                        vector[i] = ((float)random.NextDouble()-0.6f); 
                    break;
                    case "biasSoftmax":
                        vector2[i] = ((float)random.NextDouble()-0.6f);
                    return vector2;
                    case "biasReset":
                        vector[i] = -1;
                    break;
                }
            }
            return vector;
        }
        private float[,] generarMatriz(string tipo){
            Random random = new Random();
            float[,] peso;
                switch (tipo){
                    case "Compuertas": 
                        peso = new float[longitud_Xt,longitud_Xt]; 
                        for(int i = 0; i < longitud_Xt; i++){
                            for(int j = 0; j < longitud_Xt; j++){
                                peso[i, j] =  ((float)random.NextDouble()-0.4f)*0.1f;           
                            }
                        }
                    return peso;
                    case "CapaSalida":
                        peso = new float[longitud_Diccionario,longitud_Xt]; 
                        for(int i = 0; i < longitud_Diccionario; i++){
                            for(int j = 0; j < longitud_Xt; j++){
                                peso[i, j] =  ((float)random.NextDouble()-0.4f)*0.1f;           
                            }
                        }
                    return peso;
                }
                return null;
        }
        public RedNeuronalRecurrente retornarEstado(){
            return (RedNeuronalRecurrente)this.MemberwiseClone();
        }
        private void cargarRed(){
            inicializarRed();
            if (File.Exists(ruta_pesosred)){
                int p = 0;
                var fileContent = File.ReadAllText(ruta_pesosred);
                var array = fileContent.Split((string[])null, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < this.Wc.GetLength(0); i++)
                    for (int j = 0; j < this.Wc.GetLength(1); j++)
                        this.Wc[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.Wz.GetLength(0); i++)
                    for (int j = 0; j < this.Wz.GetLength(1); j++)
                        this.Wz[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.Wr.GetLength(0); i++)
                    for (int j = 0; j < this.Wr.GetLength(1); j++)
                        this.Wr[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.Uc.GetLength(0); i++)
                    for (int j = 0; j < this.Uc.GetLength(1); j++)
                        this.Uc[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.Uz.GetLength(0); i++)
                    for (int j = 0; j < this.Uz.GetLength(1); j++)
                        this.Uz[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.Ur.GetLength(0); i++)
                    for (int j = 0; j < this.Ur.GetLength(1); j++)
                        this.Ur[i,j] = float.Parse(array[p++]);
                for (int i = 0; i < this.V.GetLength(0); i++)
                    for (int j = 0; j < this.V.GetLength(1); j++)
                        this.V[i,j] = float.Parse(array[p++]);            
                for (int i = 0; i < this.biasC.Length - 1; i++)
                        this.biasC[i] = float.Parse(array[p++]);
                for (int i = 0; i < this.biasR.Length - 1; i++)
                        this.biasR[i] = float.Parse(array[p++]);
                for (int i = 0; i < this.biasV.Length - 1; i++)
                        this.biasV[i] = float.Parse(array[p++]);
                for (int i = 0; i < this.biasZ.Length - 1; i++)
                        this.biasZ[i] = float.Parse(array[p++]);
            }else Console.WriteLine("No existe el documento con el dataset de entrenamiento");
        }
        public float[,] getWz(){ return this.Wz;}
        public float [,] getWr(){return this.Wr;}
        public float [,] getWc(){return this.Wc;}
        public float [,] getUz(){return this.Uz;}
        public float [,] getUr(){return this.Ur;}
        public float [,] getUc(){return this.Uc;}
        public float [,] getV(){return this.V;}
        public float[] getbiasZ(){return this.biasZ;}
        public float[] getbiasR(){return this.biasR;}
        public float[] getbiasC(){return this.biasC;}
        public float[] getbiasV(){return this.biasV;}
        public float[] getH(){return this.H;} // h(t) = tanh(Uc X(t) + Wc s(t-1))
        public float[] getX(){return this.X;} // entrada
        public float[] getZ(){return this.Z;} // z(t) puerta
        public float[] getR(){return this.R;} // r(t) puerta
        public float[] getC(){return this.C;} // s(t) 
        public void setWz(float[,] nuevo){ this.Wz = nuevo;}
        public void setWr(float[,] nuevo){this.Wr = nuevo;}
        public void setWc(float[,] nuevo){this.Wc = nuevo;}
        public void setUz(float[,] nuevo){this.Uz = nuevo;}
        public void setUr(float[,] nuevo){this.Ur = nuevo;}
        public void setUc(float[,] nuevo){this.Uc = nuevo;}
        public void setV(float[,] nuevo){this.V = nuevo;}
        public void setbiasZ(float[] nuevo){ this.biasZ = nuevo;}
        public void setbiasR(float[] nuevo){ this.biasR = nuevo;}
        public void setbiasC(float[] nuevo){ this.biasC = nuevo;}
        public void setbiasV(float[] nuevo){ this.biasV = nuevo;}
    }
}