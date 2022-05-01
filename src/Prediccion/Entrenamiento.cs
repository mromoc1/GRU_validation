using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace DISTO_DMH_SW2
{
    class Entrenamiento
    {
        private string ruta_dataset_entrenamiento = @"./src/Data/dataset_entrenamiento.txt";
        private string ruta_pesosred = @"./src/Data/pesosred.txt";
        private Logica logica = new Logica();
        private FuncionesActivacion funciones = new FuncionesActivacion();
        private RedNeuronalRecurrente GRU;
        private RedNeuronalRecurrente GRUaux;
        private const float factorApendizaje = 0.03f;
        private const float errorMinimo = 0.2f;
        private const int cantidadEpocas = 200;
        public Entrenamiento(RedNeuronalRecurrente G)
        {
            GRU = G;
            GRUaux = GRU.retornarEstado();
        }
        public void Entrenar(RegistroVectores registroVectores)
        {
            List<float[]> listaEntradas;
            List<float[]> listaSalidasEsperadas;
            List<float[]> listaSalidasObtenidas;
            List<string> listaDatasetEntrenamiento = cargarDatosEntrenamiento();
            float errorPromedioDataSet = 90;
            for (int epoca = 1; epoca <= cantidadEpocas && errorPromedioDataSet > errorMinimo; epoca++)
            {//iterar por epocas
                errorPromedioDataSet = 0;
                Console.WriteLine("-------------------Epoca:" + epoca + "----------------------");
                foreach (string oracion in listaDatasetEntrenamiento)
                {
                    listaEntradas = new List<float[]>();
                    listaSalidasEsperadas = new List<float[]>();
                    listaSalidasObtenidas = new List<float[]>();
                    string[] oracionSplit = oracion.Split();
                    for (int i = 0; i < oracionSplit.Length - 1; i++)
                    {
                        listaEntradas.Add(registroVectores.getVector(oracionSplit[i])); // add vector de la palabra entrada
                        listaSalidasEsperadas.Add(registroVectores.getVectorOneHot(oracionSplit[i + 1])); //vector de la palabra predicha one hot
                    }
                    float errorOracion = 0;
                    List<RedNeuronalRecurrente> redDesplegada = new List<RedNeuronalRecurrente>();
                    for (int i = 0; i < listaEntradas.Count; i++)
                    {
                        float[] vectorObtenido = GRU.feedForward(listaEntradas[i]);
                        // testing entrenamiento
                        string palabra = registroVectores.getPalabra(registroVectores.getVector(Array.IndexOf(vectorObtenido, vectorObtenido.Max())));
                        //Console.WriteLine("Entrada: " + registroVectores.getPalabra(listaEntradas[i]));
                        //Console.WriteLine("Salida Obtenida: " + palabra);
                        //testing entrenamiento
                        listaSalidasObtenidas.Add(vectorObtenido);
                        errorOracion += calcularError(vectorObtenido, listaSalidasEsperadas[i]);//va acumulando el error de cada palabra de la oracion
                        redDesplegada.Add(GRU.retornarEstado()); //va guardando cada estado de la GRU en cada tiempo
                        TBPTT(redDesplegada, listaSalidasObtenidas, listaSalidasEsperadas);
                    }
                    errorOracion = errorOracion / listaEntradas.Count; // promedio de error de la oracion
                    // Console.WriteLine("Error Oracion:" + errorOracion);
                    errorPromedioDataSet += errorOracion; // va a cumulando el error de cada oracion
                    // TBPTT(redDesplegada);
                }
                errorPromedioDataSet = errorPromedioDataSet / listaDatasetEntrenamiento.Count; // promedio de error del dataset
                Console.WriteLine("Error Promedio Dataset:" + errorPromedioDataSet);
                if ((epoca == cantidadEpocas && errorPromedioDataSet > 0.4) || (errorPromedioDataSet != 30))
                {
                    Console.WriteLine("RESET");
                    GRU = new RedNeuronalRecurrente(false, 300, 300);
                    Entrenar(registroVectores);
                }
            }
            guardarRed(GRU);
        }
        private void TBPTT(List<RedNeuronalRecurrente> redDesplegada, List<float[]> listaSalidasObtenidas, List<float[]> listaSalidasEsperadas)
        {
            float[] dL_dh = new float[300];
            float[] dL_dc = new float[300];
            float[] dL_dr = new float[300];
            float[] dL_dz = new float[300];
            float[,] dL_dV = new float[300, 300];
            float[,] dL_dWr = new float[300, 300];
            float[,] dL_dWz = new float[300, 300];
            float[,] dL_dWc = new float[300, 300];
            float[,] dL_dUr = new float[300, 300];
            float[,] dL_dUz = new float[300, 300];
            float[,] dL_dUc = new float[300, 300];
            for (int i = redDesplegada.Count - 1; i >= 0; i--)
            {
                if (i > 0)
                {
                    dL_dh = calculardL_dh(listaSalidasObtenidas[i], listaSalidasEsperadas[i], redDesplegada[i].getV());
                    dL_dc = calculardL_dc(dL_dh, redDesplegada[i].getZ());
                    dL_dr = calculardL_dr(dL_dr, redDesplegada[i].getWc(), redDesplegada[i - 1].getH(), redDesplegada[i].getUc(), redDesplegada[i].getX(), redDesplegada[i].getR());
                    dL_dz = calculardL_dz(dL_dh, redDesplegada[i - 1].getH(), redDesplegada[i].getC());
                    dL_dV = sumaV(dL_dV, listaSalidasObtenidas[i], listaSalidasEsperadas[i], redDesplegada[i].getH());
                    dL_dWr = sumaWr_Wz_Ur_Uz(dL_dWr, dL_dr, redDesplegada[i].getR(), redDesplegada[i - 1].getH());
                    dL_dWz = sumaWr_Wz_Ur_Uz(dL_dWz, dL_dz, redDesplegada[i].getZ(), redDesplegada[i - 1].getH());
                    dL_dWc = sumaWc(dL_dWc, dL_dc, redDesplegada[i].getC(), redDesplegada[i].getR(), redDesplegada[i - 1].getH());
                    dL_dUr = sumaWr_Wz_Ur_Uz(dL_dUr, dL_dr, redDesplegada[i].getR(), redDesplegada[i].getX());
                    dL_dUz = sumaWr_Wz_Ur_Uz(dL_dUz, dL_dz, redDesplegada[i].getZ(), redDesplegada[i].getX());
                    dL_dUc = sumaUc(dL_dUc, dL_dc, redDesplegada[i].getC(), redDesplegada[i].getX());
                }
                else if (i == 0)
                {
                    dL_dh = calculardL_dh(listaSalidasObtenidas[i], listaSalidasEsperadas[i], redDesplegada[i].getV());
                    dL_dc = calculardL_dc(dL_dh, redDesplegada[i].getZ());
                    dL_dr = calculardL_dr(dL_dr, redDesplegada[i].getWc(), redDesplegada[i].getH(), redDesplegada[i].getUc(), redDesplegada[i].getX(), redDesplegada[i].getR());
                    dL_dz = calculardL_dz(dL_dh, redDesplegada[i].getH(), redDesplegada[i].getC());
                    dL_dV = sumaV(dL_dV, listaSalidasObtenidas[i], listaSalidasEsperadas[i], redDesplegada[i].getH());
                    dL_dWr = sumaWr_Wz_Ur_Uz(dL_dWr, dL_dr, redDesplegada[i].getR(), redDesplegada[i].getH());
                    dL_dWz = sumaWr_Wz_Ur_Uz(dL_dWz, dL_dz, redDesplegada[i].getZ(), redDesplegada[i].getH());
                    dL_dWc = sumaWc(dL_dWc, dL_dc, redDesplegada[i].getC(), redDesplegada[i].getR(), redDesplegada[i].getH());
                    dL_dUr = sumaWr_Wz_Ur_Uz(dL_dUr, dL_dr, redDesplegada[i].getR(), redDesplegada[i].getX());
                    dL_dUz = sumaWr_Wz_Ur_Uz(dL_dUz, dL_dz, redDesplegada[i].getZ(), redDesplegada[i].getX());
                    dL_dUc = sumaUc(dL_dUc, dL_dc, redDesplegada[i].getC(), redDesplegada[i].getX());
                }

            }
            GRU = GRUaux.retornarEstado();
            GRU.setV(calcularNuevoPesos(dL_dV, GRU.getV()));
            GRU.setWr(calcularNuevoPesos(dL_dWr, GRU.getWr()));
            GRU.setWz(calcularNuevoPesos(dL_dWz, GRU.getWz()));
            GRU.setWc(calcularNuevoPesos(dL_dWc, GRU.getWc()));
            GRU.setUr(calcularNuevoPesos(dL_dUr, GRU.getUr()));
            GRU.setUz(calcularNuevoPesos(dL_dUz, GRU.getUz()));
            GRU.setUc(calcularNuevoPesos(dL_dUc, GRU.getUc()));
            GRUaux = GRU.retornarEstado();
        }
        private float[,] calcularNuevoPesos(float[,] matrizGrad, float[,] matriz)
        {
            float[,] mult = new float[matriz.GetLength(0), matriz.GetLength(1)];
            for (int i = 0; i < matriz.GetLength(0); i++)
                for (int j = 0; j < matriz.GetLength(1); j++)
                    mult[i, j] = -1 * factorApendizaje * matrizGrad[i, j];
            return logica.sumaMatrices(matriz, mult);
        }
        private float[,] sumaUc(float[,] dL_dUc, float[] dL_dc, float[] C, float[] X)
        {
            float[] cuadrado = new float[C.Length];
            for (int i = 0; i < C.Length; i++)
                cuadrado[i] = C[i] * C[i];
            float[] resta = logica.resta(cuadrado);
            float[] hadam1 = logica.productoHadamard(resta, dL_dc);
            float[,] mult = logica.multiplicarVectores(hadam1, X);
            return logica.sumaMatrices(dL_dUc, mult);
        }
        private float[,] sumaWc(float[,] dL_dWc, float[] dL_dc, float[] C, float[] R, float[] Hanterior)
        {
            float[] cuadrado = new float[C.Length];
            for (int i = 0; i < C.Length; i++)
                cuadrado[i] = C[i] * C[i];
            float[] resta = logica.resta(cuadrado);
            float[] hadam1 = logica.productoHadamard(resta, R);
            float[] hadam2 = logica.productoHadamard(hadam1, dL_dc);
            float[] hadam3 = logica.productoHadamard(R, Hanterior);
            float[,] mult = logica.multiplicarVectores(hadam3, hadam2);
            return logica.sumaMatrices(dL_dWc, mult);
        }
        private float[,] sumaWr_Wz_Ur_Uz(float[,] dL_dWr, float[] dL_dr, float[] R, float[] Hanterior)
        {
            float[] resta = logica.resta(R);
            float[] hadam1 = logica.productoHadamard(resta, R);
            float[] hadam2 = logica.productoHadamard(hadam1, dL_dr);
            float[,] mult = logica.multiplicarVectores(hadam2, Hanterior);
            return logica.sumaMatrices(dL_dWr, mult);
        }
        private float[,] sumaV(float[,] dL_dV, float[] salidaObtenida, float[] salidaEsperada, float[] H)
        {
            float[] resta = logica.restaVectores(salidaObtenida, salidaEsperada);
            float[,] mult = logica.multiplicarVectores(resta, H);
            return logica.sumaMatrices(mult, dL_dV);
        }
        private float[] calculardL_dz(float[] dL_dh, float[] Hanterior, float[] Ct)
        {
            float[] resta = logica.restaVectores(Hanterior, Ct);
            return logica.productoHadamard(dL_dh, resta);
        }
        private float[] calculardL_dr(float[] dL_dr, float[,] Wc, float[] Hanterior, float[,] Uc, float[] Xt, float[] Rt)
        {
            float[] WcporHanterior = logica.multiplcarMatrizVector(Wc, Hanterior);
            float[] UcporXt = logica.multiplcarMatrizVector(Uc, Xt);
            float[] suma1 = logica.sumaVectores(WcporHanterior, UcporXt);
            float[] suma2 = logica.sumaVectores(suma1, Rt);
            for (int i = 0; i < suma2.Length; i++)
                suma2[i] = (float)(1 - Math.Pow(Math.Tanh(suma2[i]), 2));
            float[] hadam1 = logica.productoHadamard(WcporHanterior, suma2);
            return logica.productoHadamard(dL_dr, hadam1);
        }
        private float[] calculardL_dc(float[] dL_dh, float[] Z)
        {
            float[] resta = logica.resta(Z);
            return logica.productoHadamard(dL_dh, resta);
        }
        private float[] calculardL_dh(float[] salidaObtenida, float[] salidaEsperada, float[,] V)
        {
            float[] resta = logica.restaVectores(salidaObtenida, salidaEsperada);
            return logica.multiplcarMatrizVector(V, resta); // posible vector * matriz
        }
        public float calcularError(float[] vectorObtenido, float[] vectorEsperado)
        {
            float error = 0;
            for (int i = 0; i < vectorEsperado.Length; i++)
            {
                error += vectorEsperado[i] * (float)Math.Log(vectorObtenido[i]);
            }
            return -error;
        }
        private List<string> cargarDatosEntrenamiento()
        {
            List<String> listaDatasetEntrenamiento = new List<string>();
            if (File.Exists(ruta_dataset_entrenamiento))
            {
                IEnumerable<string> lines = File.ReadLines(ruta_dataset_entrenamiento);
                for (int i = 0; i < lines.Count(); i++)
                {
                    listaDatasetEntrenamiento.Add(lines.ElementAt(i));
                }
            }
            return listaDatasetEntrenamiento;
        }
        private void guardarRed(RedNeuronalRecurrente red)
        {
            if (System.IO.File.Exists(ruta_pesosred))
            {
                System.IO.FileStream f = System.IO.File.Create(ruta_pesosred);
                f.Close();
            }
            using (System.IO.StreamWriter sw = System.IO.File.AppendText(ruta_pesosred))
            {
                for (int i = 0; i < red.getWc().GetLength(0); i++)
                    for (int j = 0; j < red.getWc().GetLength(1); j++)
                        sw.Write(red.getWc()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getWz().GetLength(0); i++)
                    for (int j = 0; j < red.getWz().GetLength(1); j++)
                        sw.Write(red.getWz()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getWr().GetLength(0); i++)
                    for (int j = 0; j < red.getWr().GetLength(1); j++)
                        sw.Write(red.getWr()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getUc().GetLength(0); i++)
                    for (int j = 0; j < red.getUc().GetLength(1); j++)
                        sw.Write(red.getUc()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getUz().GetLength(0); i++)
                    for (int j = 0; j < red.getUz().GetLength(1); j++)
                        sw.Write(red.getUz()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getUr().GetLength(0); i++)
                    for (int j = 0; j < red.getUr().GetLength(1); j++)
                        sw.Write(red.getUr()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getV().GetLength(0); i++)
                    for (int j = 0; j < red.getV().GetLength(1); j++)
                        sw.Write(red.getV()[i, j] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getbiasC().Length - 1; i++)
                    sw.Write(red.getbiasC()[i] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getbiasR().Length - 1; i++)
                    sw.Write(red.getbiasR()[i] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getbiasV().Length - 1; i++)
                    sw.Write(red.getbiasV()[i] + " ");
                sw.WriteLine();
                for (int i = 0; i < red.getbiasZ().Length - 1; i++)
                    sw.Write(red.getbiasZ()[i] + " ");
                sw.Close();
            }
        }
    }
}
