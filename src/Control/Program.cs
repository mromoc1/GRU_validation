using System;
namespace DISTO_DMH_SW2
{
    class Program
    {
        public static void Main(string[] args)
        {
            Control control = new Control();
            String op = "";
            do
            {
                // System.Console.Clear();
                Console.WriteLine("Este es un prototipo de prueba del software DISTO-DMH");
                Console.WriteLine("Seleccione una opcion:");
                Console.WriteLine("1.. Entrenar");
                Console.WriteLine("2.. Iniciar Prueba de prediccion");
                Console.WriteLine("3.. Salir");
                op = Console.ReadLine();
                switch (op)
                {
                    case "1":
                        Console.Clear();
                        control.entrenarSistema();
                        break;
                    case "2":
                        Console.Clear();
                        control.predecir();
                        break;
                    default:
                        Console.WriteLine("Opcion invalida");
                        break;
                }
            } while (op != "3");
        }
    }
}
