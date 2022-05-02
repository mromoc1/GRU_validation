using System;
using System.Linq;
namespace DISTO_DMH_SW2
{
    public class Control
    {
        private RegistroVectores registroVectores;
        private RedNeuronalRecurrente GRU;
        private Entrenamiento entrenamiento;
        private Boolean cargarRed = true; // false para empezar una red de 0
        public Control()
        {
            registroVectores = new RegistroVectores();
            GRU = new RedNeuronalRecurrente(cargarRed, registroVectores.longitudEntrada(), registroVectores.longitudDiccionario());
            entrenamiento = new Entrenamiento(GRU.retornarEstado());
        }
        public void entrenarSistema()
        {
            entrenamiento.Entrenar(registroVectores);
        }
        public void predecir()
        {
            char op = ' ';
            string a = "";
            do
            {
                Console.WriteLine("Seleccione una opcion:");
                Console.WriteLine("1.. Prueba de prediccion");
                Console.WriteLine("0.. salir de la prueba");
                op = Console.ReadKey().KeyChar;
                switch (op)
                {
                    case '1': //Ingresar frase
                        float[] vectorSalida = { };
                        Console.WriteLine("Ingrese una frase:");
                        a = Console.ReadLine();
                        string[] oracionSplit = a.Split();
                        for (int i = 0; i <= oracionSplit.Length - 1; i++)
                        {
                            vectorSalida = GRU.feedForward(registroVectores.getVector(oracionSplit[i]));
                        }
                        string palabra = registroVectores.getPalabra(registroVectores.getVector(Array.IndexOf(vectorSalida, vectorSalida.Max())));
                        Console.WriteLine("El sistema dice: " + palabra);
                        Console.WriteLine("");
                        break;
                    default:
                        Console.WriteLine("Opcion invalida");
                        break;
                }
            } while (op != '0');
        }
    }
}

