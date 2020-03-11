using System;
using System.Collections.Generic;
using System.Linq;
using Windows.Foundation;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.Storage;
using Windows.UI.Xaml.Media.Imaging;
using Windows.AI.MachineLearning;
using Windows.Media;
using Windows.Storage.Streams;
using System.Threading.Tasks;
using MNIST_DemoML.Model;
using Microsoft.ML;

namespace MNIST_Demo
{
    public sealed partial class MainPage : Page
    {
        private mnistModel modelGen;
        private mnistInput mnistInput = new mnistInput();
        private mnistOutput mnistOutput;
        //private LearningModelSession    session;
        private Helper helper = new Helper();
        RenderTargetBitmap renderBitmap = new RenderTargetBitmap();

        public MainPage()
        {
            this.InitializeComponent();

            // Set supported inking device types.
            inkCanvas.InkPresenter.InputDeviceTypes = Windows.UI.Core.CoreInputDeviceTypes.Mouse | Windows.UI.Core.CoreInputDeviceTypes.Pen | Windows.UI.Core.CoreInputDeviceTypes.Touch;
            inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(
                new Windows.UI.Input.Inking.InkDrawingAttributes()
                {
                    Color = Windows.UI.Colors.White,
                    Size = new Size(22, 22),
                    IgnorePressure = true,
                    IgnoreTilt = true,
                }
            );
            LoadModelAsync();
        }

        private async Task LoadModelAsync()
        {
            //Load a machine learning model
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
            modelGen = await mnistModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);
        }

        private async void recognizeButton_Click(object sender, RoutedEventArgs e)
        {
            //Bind model input with contents from InkCanvas
            StorageFile vf = await helper.GetHandWrittenImage(inkGrid);

            //Evaluate the model
            var prediction = await Predict(new ModelInput { ImageSource = vf.Path });

            numberLabel.Text = prediction.Prediction;

            await vf.DeleteAsync();
        }

        private async Task<ModelOutput> Predict(ModelInput input)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Register NormalizeMapping
            mlContext.ComponentCatalog.RegisterAssembly(typeof(NormalizeMapping).Assembly);

            // Register LabelMapping
            mlContext.ComponentCatalog.RegisterAssembly(typeof(LabelMapping).Assembly);

            // Load model & create prediction engine
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/MLModel.zip"));
            //string modelPath = @"C:\Users\luquinta.REDMOND\AppData\Local\Temp\MLVSTools\MNIST_DemoML\MNIST_DemoML.Model\MLModel.zip";
            ITransformer mlModel = mlContext.Model.Load(modelFile.Path, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use model to make prediction on input data
            ModelOutput result = predEngine.Predict(input);
            return result;
        }

        private void clearButton_Click(object sender, RoutedEventArgs e)
        {
            inkCanvas.InkPresenter.StrokeContainer.Clear();
            numberLabel.Text = "";
        }
    }
}
