using System;
using System.Collections.Generic;
using System.IO;
namespace DISTO_DMH_SW2{
    public class RegistroVectores{
        private string ruta_dataset_palabrasvectorizadas = @"./src/Data/dataset_palabrasvectorizadas.txt";
        private Dictionary<string, float[]> registroStringVector;
        private Dictionary<float[], string> registroVectorString;
        private Dictionary<int, float[]> registroIndexVector;
        private Dictionary<string, float[]> registroOneHot;
        private int longitud_vector = 300; //limitado por el dataset esperado
        public RegistroVectores(){
            cargarRegistro(); 
        }
        public float[] getVector(int index){
            return registroIndexVector[index];
        }
        public float[] getVector(string palabra){
            if (registroStringVector.ContainsKey(palabra.ToLower())){
                return registroStringVector[palabra.ToLower()];
            }else{
                Console.WriteLine("No se encontró la palabra " +palabra);
                return registroStringVector["eso"];
            }
        }
        public float[] getVectorOneHot(string palabra){
            return registroOneHot[palabra.ToLower()];
        }
        public string getPalabra(float[] vector){
            return registroVectorString[vector];
        }
        private void cargarRegistro(){
            registroStringVector = new Dictionary<string, float[]>();
            registroVectorString = new Dictionary<float[], string>();
            registroOneHot = new Dictionary<string, float[]>();
            registroIndexVector = new Dictionary<int, float[]>();
            if (File.Exists(ruta_dataset_palabrasvectorizadas)){
                var fileContent = File.ReadAllText(ruta_dataset_palabrasvectorizadas);
                var array = fileContent.Split((string[])null, StringSplitOptions.RemoveEmptyEntries);
                float[] arregloAux = new float[longitud_vector];
                float n;
                int j = 0;
                int numero = 0;
                for (int i = 0; i < array.Length; i++){
                    if (float.TryParse(array[i], out n)){
                        arregloAux[j] = n;
                    }else{
                        arregloAux = new float[longitud_vector];
                        j = -1;
                        if (!registroStringVector.ContainsKey(array[i].ToLower())){
                            registroStringVector.Add(array[i].ToLower(), arregloAux);
                            registroVectorString.Add(arregloAux, array[i].ToLower());
                            registroIndexVector.Add(numero, arregloAux);
                            numero++;
                        }else
                        {
                            Console.WriteLine(array[i].ToLower());
                            Console.ReadKey();
                        }
                    }j++;
                }
                for(int i = 0 ; i < registroIndexVector.Count; i++)
                    registroOneHot.Add(registroVectorString[registroIndexVector[i]],crearVectorOneHot(i,registroIndexVector.Count));
                Console.WriteLine("Registro Vectores cargado exitosamente "+registroStringVector.Count);
            }else Console.WriteLine("NO EXISTE EL datasetPalabrasEspanol");
        }
        private float[] crearVectorOneHot(int pos, int longitud){
            float[] onehot = new float[longitud];
            for (int i = 0; i < longitud; i++){
                if (i == pos){ onehot[i] = 1; }
                else{ onehot[i] = 0; }}
            return onehot;
        }
        public int longitudDiccionario(){
            return registroOneHot.Count;
        }
        public int longitudEntrada(){
            return longitud_vector;
        }
    }
}
