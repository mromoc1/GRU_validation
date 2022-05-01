using System;
namespace DISTO_DMH_SW2{
    class FuncionesActivacion{
        public float[] Sigmoide(float[] suma){
            float[] v = new float[suma.Length];
            for (int i = 0; i < suma.Length; i++){
                v[i] = (float)(1 / (1 + Math.Exp(-suma[i])));
            }
            return v;
        }        
        public float[] tangenteHiperbolica(float[] x){
            float[] y = new float[x.Length];
            for (int i = 0; i < x.Length; i++){
                y[i] = (float)Math.Tanh(x[i]);
            }
            return y;
        }
        public float[] softmax (float[] inputSoftmax){
            float[] res = new float[inputSoftmax.Length];
            double suma = 0;
            for(int i = 0 ; i<inputSoftmax.Length; i++){
                suma += Math.Exp(inputSoftmax[i]);
            } 
            for(int j=0; j< inputSoftmax.Length; j++){
                res[j] = (float)(Math.Exp(inputSoftmax[j])/suma);
            }
            return res;
        }
    }
}
