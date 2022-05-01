using System;
using System.Linq;
namespace DISTO_DMH_SW2
{
    public class Control
    {
        private RegistroVectores registroVectores;
        private RedNeuronalRecurrente GRU;
        private Entrenamiento entrenamiento;
        private Boolean cargarRed = false; // false para empezar una red de 0
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
            string a = " ";
            do
            {
                Console.Clear();
                Console.WriteLine("Este es un prototipo de prueba del software DISTO-DMH");
                Console.WriteLine("Seleccione una opcion:");
                Console.WriteLine("1.. Prueba de prediccion");
                Console.WriteLine("0.. Iniciar Prueba de prediccion");
                op = Console.ReadKey().KeyChar;
                switch (op)
                {
                    case '1': //Ingresar frase
                        float[] vectorSalida = { };
                        Console.Clear();
                        Console.WriteLine("Ingrese una frase:");
                        a = Console.ReadLine();
                        string[] oracionSplit = a.Split();
                        for (int i = 0; i < oracionSplit.Length - 1; i++)
                        {
                            vectorSalida = GRU.feedForward(registroVectores.getVector(oracionSplit[i]));
                        }
                        string palabra = registroVectores.getPalabra(registroVectores.getVector(Array.IndexOf(vectorSalida, vectorSalida.Max())));
                        Console.WriteLine("El sistema dice: " + palabra);
                        break;
                    default:
                        Console.WriteLine("Opcion invalida");
                        break;
                }
            } while (op != '0');
        }
    }
}

