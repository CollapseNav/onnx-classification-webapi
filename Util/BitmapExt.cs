using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

public static class BitmapExt
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] StdDev = [0.229f, 0.224f, 0.225f];
    public static SKBitmap Resize(this SKBitmap bitmap, int width, int height)
    {
        SKBitmap resizedBitmap = new(width, height, false);
        SKCanvas sKCanvas = new SKCanvas(resizedBitmap);
        SKRect sKRect = new SKRect(0, 0, width, height);
        sKCanvas.DrawImage(SKImage.FromBitmap(bitmap), sKRect);
        return resizedBitmap;
    }

    public static List<NamedOnnxValue> Preprocess(this SKBitmap bitmap, InferenceSession inf)
    {
        var inputName = inf.InputNames[0];
        var inputMetadata = inf.InputMetadata[inputName];
        var dimensions = inputMetadata.Dimensions;
        dimensions[0] = 1;
        var tensor = new DenseTensor<float>(dimensions);
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                var pixel = bitmap.GetPixel(x, y);
                byte blue = pixel.Blue;
                byte green = pixel.Green;
                byte red = pixel.Red;
                tensor[0, 0, y, x] = ((red / 255f) - Mean[0]) / StdDev[0];
                tensor[0, 1, y, x] = ((green / 255f) - Mean[1]) / StdDev[1];
                tensor[0, 2, y, x] = ((blue / 255f) - Mean[2]) / StdDev[2];
            }
        }
        return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
    }
}