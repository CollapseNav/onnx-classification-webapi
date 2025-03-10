using Collapsenav.Net.Tool;
using SkiaSharp;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.MapPost("/Get_Result", async (HttpRequest request) =>
{
    var file = request.Form.Files[0];
    using MemoryStream ms = new MemoryStream();
    await file.OpenReadStream().CopyToAsync(ms);
    ms.SeekToOrigin();
    var inf = ModelLoader.LoadModel("onnx_classification_webapi.Model.mobilenetv2-10.onnx");
    var labels = ModelLoader.LoadLabels("onnx_classification_webapi.Model.label_cn.txt");
    SKData sKData = SKData.Create(ms);
    SKBitmap sKBitmap = SKBitmap.Decode(sKData);
    var inputName = inf.InputNames[0];
    var inputMatadata = inf.InputMetadata[inputName];
    var dimensions = inputMatadata.Dimensions;
    dimensions[0] = 1;
    var resizeBitmap = sKBitmap.Resize(dimensions[2], dimensions[3]);
    var inputs = resizeBitmap.Preprocess(inf);
    using var results = inf.Run(inputs);
    var outputs = SoftMax(results[0].AsEnumerable<float>());
    var result = outputs.Select((x, i) => new { x, label = labels[i] })
                    .OrderByDescending(x => x.x)
                    .Take(6)
                    .ToArray();
    return result;

    IEnumerable<float> SoftMax(IEnumerable<float> output)
    {
        float sum = output.Sum(item => (float)Math.Exp(item));
        return output.Select(item => (float)Math.Exp(item) / sum);
    }
}).DisableAntiforgery();
app.Run();

