namespace DISTO_DMH_SW2{
    class Logica{
        public float[,] multiplicarVectores(float[] vec1, float[] vec2){
            float[,] matriz = new float[vec1.Length,vec2.Length];
            for (int i = 0; i < vec1.Length; i++){
                for (int j = 0; j < vec2.Length; j++){
                    matriz[i,j] = vec1[i] * vec2[j];
                }
            }
            return matriz;
        }
        public float[] sumaVectores(float[] wx, float[] uh){
            float[] vectorFinal = new float[wx.Length];
            for (int i = 0; i < wx.Length; i++){
                vectorFinal[i] = wx[i] + uh[i];
            }
            return vectorFinal;
        }
        public float[,] sumaMatrices(float[,] m1, float[,] m2){
            float[,] res = new float[m1.GetLength(0),m1.GetLength(1)];
            for(int i = 0;i < m1.GetLength(0); i++){
                for(int j = 0; j < m1.GetLength(1); j++){
                    res[i, j] = m1[i, j] + m2[i, j];
                }
            }
            return res;
        }
        public float[] resta(float[] zt){
            float[] resta = new float[zt.Length];
            for(int i = 0; i < zt.Length; i++){
                resta[i] = 1 - zt[i]; 
            }
            return resta;
        }
        public float[] multiplcarMatrizVector(float[,] w, float[] x){
            float[] vectorFinal = new float[w.GetLength(0)];
            for (int fila = 0; fila < w.GetLength(0); fila++){
                for (int col = 0; col < w.GetLength(1); col++){
                    vectorFinal[fila] += w[fila,col] * x[col];
                }
            }
            return vectorFinal;
        }
        public float[] restaVectores(float[] v1, float[] v2){
            float[] vf = new float[v1.Length];
            for(int i = 0; i< v1.Length; i++ ){
                vf[i] = v1[i] - v2[i];
            }
            return vf;
        }
        public float[] productoHadamard(float[] h, float[] rt){
            float[] v = new float[h.Length];
            for (int i = 0; i < v.Length; i++){
                v[i] = h[i] * rt[i];
            }
            return v;
        }
    }
}
