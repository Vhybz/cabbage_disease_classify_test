import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'tflite_service_interface.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:flutter/foundation.dart';

class TFLiteService implements TFLiteServiceInterface {
  Interpreter? _interpreter;
  List<String> _labels = [
    'Alternaria Leaf Spot',
    'Bacterial Spot Rot',
    'Black Rot',
    'Cabbage Aphid Infestation',
    'Downy Mildew',
    'Healthy',
    'Not a Cabbage Leaf',
    'Club Root',
    'Ring Spot',
  ];

  @override
  Future<void> loadModel() async {
    if (_interpreter != null) return;
    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        'assets/model/cabbage_float32.tflite',
        options: options,
      );
      _interpreter!.allocateTensors();
      
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      debugPrint('TFLite Model Metadata: Input Shape=${inputTensor.shape}, Type=${inputTensor.type}; Output Shape=${outputTensor.shape}');

      try {
        final labelsData = await rootBundle.loadString('assets/model/cabbage_labels.txt');
        _labels = labelsData.split('\n').where((s) => s.isNotEmpty).map((s) => s.trim()).toList();
        debugPrint('TFLite: Labels loaded: $_labels');
      } catch (e) {
        debugPrint('TFLite: Using default labels: $_labels');
      }
    } catch (e) {
      debugPrint('TFLite Error (Load): $e');
    }
  }

  Future<String> _downloadImage(String url) async {
    try {
      final response = await http.get(Uri.parse(url));
      final documentDirectory = await getTemporaryDirectory();
      final file = File('${documentDirectory.path}/temp_image_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await file.writeAsBytes(response.bodyBytes);
      return file.path;
    } catch (e) {
      throw Exception('TFLite: Failed to download image: $e');
    }
  }

  @override
  Future<Map<String, dynamic>?> classifyImage(dynamic imageInput) async {
    try {
      if (_interpreter == null) await loadModel();
      if (_interpreter == null) {
        debugPrint('TFLite Mobile: Interpreter is null, falling back to Render API...');
        return await _classifyWithRenderAPI(imageInput);
      }

      Uint8List bytes;
      if (imageInput is Uint8List) {
        bytes = imageInput;
      } else if (imageInput is List<int>) {
        bytes = Uint8List.fromList(imageInput);
      } else if (imageInput is String) {
        if (imageInput.startsWith('http')) {
          final downloadedPath = await _downloadImage(imageInput);
          bytes = File(downloadedPath).readAsBytesSync();
        } else {
          final File imageFile = File(imageInput);
          if (!imageFile.existsSync()) {
            debugPrint('TFLite Error: Image file does not exist at $imageInput, trying Render API...');
            return await _classifyWithRenderAPI(imageInput);
          }
          bytes = imageFile.readAsBytesSync();
        }
      } else {
        debugPrint('TFLite Error: Unsupported image input type: ${imageInput.runtimeType}');
        return await _classifyWithRenderAPI(imageInput);
      }

      var image = img.decodeImage(bytes);
      if (image == null) {
        debugPrint('TFLite Error: Failed to decode image bytes (${bytes.length} bytes), trying Render API...');
        return await _classifyWithRenderAPI(imageInput);
      }
      image = img.bakeOrientation(image);
      debugPrint('TFLite Info: Successfully decoded image (${image.width}x${image.height})');

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      // Input dimension specified as 299x299 (NHWC RGB)
      final int inputHeight = inputTensor.shape.length > 1 ? inputTensor.shape[1] : 299;
      final int inputWidth = inputTensor.shape.length > 2 ? inputTensor.shape[2] : 299;

      img.Image resizedImage = img.copyResize(image, width: inputWidth, height: inputHeight);

      // Allocate the Float32 buffer (1 * height * width * 3)
      var inputBuffer = Float32List(1 * inputHeight * inputWidth * 3);
      int index = 0;

      // Extract channels sequentially (RGB sequence) using raw 0.0 - 255.0 floats (normalisation is baked into the model)
      for (int y = 0; y < inputHeight; y++) {
        for (int x = 0; x < inputWidth; x++) {
          var pixel = resizedImage.getPixel(x, y);
          
          inputBuffer[index++] = pixel.r.toDouble();
          inputBuffer[index++] = pixel.g.toDouble();
          inputBuffer[index++] = pixel.b.toDouble();
        }
      }

      int numClasses = outputTensor.shape.length > 1 ? outputTensor.shape[1] : _labels.length;
      var output = Float32List(numClasses).reshape([1, numClasses]);
      _interpreter!.run(inputBuffer.reshape([1, inputHeight, inputWidth, 3]), output);

      List<double> rawScores = List<double>.from(output[0]);
      
      // Ensure probabilities are proper 0.0 - 1.0 softmax values
      List<double> probabilities = _applySoftmax(rawScores);

      double maxScore = 0.0;
      int maxIndex = 0;

      debugPrint('--- AI DEBUG SCORES ---');
      for (int i = 0; i < probabilities.length && i < _labels.length; i++) {
        debugPrint('${_labels[i]}: ${(probabilities[i] * 100).toStringAsFixed(2)}%');
        if (probabilities[i] > maxScore) {
          maxScore = probabilities[i];
          maxIndex = i;
        }
      }
      debugPrint('-----------------------');

      String predictedLabel = maxIndex < _labels.length ? _labels[maxIndex] : 'Unidentified';
      bool isNotLeaf = maxIndex == 6 || predictedLabel == 'Not a Cabbage Leaf' || predictedLabel == 'Not cabbage';

      if (isNotLeaf) {
        return {
          'label': 'Not a Cabbage Leaf',
          'confidence': maxScore,
          'index': maxIndex,
          'isLeaf': false,
        };
      }

      return {
        'label': predictedLabel,
        'confidence': maxScore,
        'index': maxIndex,
        'isLeaf': true, 
        'all_scores': probabilities,
      };
    } catch (e) {
      debugPrint('TFLite Inference Error: $e. Falling back to Render API...');
      return await _classifyWithRenderAPI(imageInput);
    }
  }

  Future<Map<String, dynamic>?> _classifyWithRenderAPI(dynamic imageInput) async {
    try {
      final String apiUrl = dotenv.env['RENDER_API_URL'] ?? 'https://cabbage-disease-classify-test.onrender.com/predict';
      
      Uint8List imageBytes;
      if (imageInput is Uint8List) {
        imageBytes = imageInput;
      } else if (imageInput is List<int>) {
        imageBytes = Uint8List.fromList(imageInput);
      } else if (imageInput is String) {
        final File imageFile = File(imageInput.startsWith('http') ? await _downloadImage(imageInput) : imageInput);
        if (!imageFile.existsSync()) return null;
        imageBytes = await imageFile.readAsBytes();
      } else {
        return null;
      }

      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: 'image.jpg',
      ));

      final streamedResponse = await request.send().timeout(const Duration(seconds: 15));
      final apiResponse = await http.Response.fromStream(streamedResponse);

      if (apiResponse.statusCode == 200) {
        final dynamic data = jsonDecode(apiResponse.body);
        String label = data['disease'] ?? data['label'] ?? 'Healthy';
        double confidence = (data['confidence'] as num?)?.toDouble() ?? 0.0;
        bool isLeaf = (data['is_leaf'] as bool?) ?? (label != 'Not a Cabbage Leaf' && label != 'Not cabbage');

        if (!isLeaf || label == 'Not a Cabbage Leaf' || label == 'Not cabbage') {
          return {
            'label': 'Not a Cabbage Leaf',
            'confidence': confidence,
            'index': 6,
            'isLeaf': false,
          };
        }

        return {
          'label': label,
          'confidence': confidence,
          'index': _labels.contains(label) ? _labels.indexOf(label) : 0,
          'isLeaf': true,
        };
      }
    } catch (e) {
      debugPrint('Render API Fallback Error: $e');
    }
    return null;
  }

  List<double> _applySoftmax(List<double> logits) {
    if (logits.isEmpty) return logits;
    double sum = logits.reduce((a, b) => a + b);
    // If logits sum is not ~1.0, apply softmax to convert logits to probabilities
    if ((sum - 1.0).abs() > 0.05) {
      double maxLogit = logits.reduce((a, b) => a > b ? a : b);
      List<double> expValues = logits.map((x) => math.exp(x - maxLogit)).toList();
      double sumExp = expValues.reduce((a, b) => a + b);
      if (sumExp == 0) return logits;
      return expValues.map((x) => x / sumExp).toList();
    }
    return logits;
  }

  @override
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
