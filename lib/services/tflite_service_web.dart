import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'tflite_service_interface.dart';

class TFLiteService implements TFLiteServiceInterface {
  final List<String> _labels = [
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
    debugPrint('TFLite Service (Web): Using Render-hosted Python API.');
  }

  @override
  Future<Map<String, dynamic>?> classifyImage(dynamic imageInput) async {
    try {
      debugPrint('TFLite Web: Sending image bytes to Render API...');
      
      Uint8List imageBytes;
      if (imageInput is Uint8List) {
        imageBytes = imageInput;
      } else if (imageInput is List<int>) {
        imageBytes = Uint8List.fromList(imageInput);
      } else if (imageInput is String) {
        if (imageInput.startsWith('http') || imageInput.startsWith('blob:') || imageInput.startsWith('data:')) {
          final response = await http.get(Uri.parse(imageInput));
          imageBytes = response.bodyBytes;
        } else {
          final response = await http.get(Uri.parse(imageInput));
          imageBytes = response.bodyBytes;
        }
      } else {
        throw Exception('Invalid web image input type: ${imageInput.runtimeType}');
      }

      // Call Render API with automatic retry while container wakes from cold-start
      final String apiUrl = dotenv.env['RENDER_API_URL'] ?? 'https://cabbage-disease-classify-test.onrender.com/predict';
      
      int attempts = 0;
      const int maxAttempts = 12; // Wait up to ~3 minutes for server response

      while (attempts < maxAttempts) {
        attempts++;
        try {
          debugPrint('TFLite Web: Sending request to Render API (Attempt $attempts)...');
          var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
          request.files.add(http.MultipartFile.fromBytes(
            'file',
            imageBytes,
            filename: 'image.jpg',
          ));

          final streamedResponse = await request.send().timeout(const Duration(seconds: 25));
          final apiResponse = await http.Response.fromStream(streamedResponse);

          if (apiResponse.statusCode == 200) {
            final dynamic data = jsonDecode(apiResponse.body);
            String label = data['disease'] ?? data['label'] ?? 'Healthy';
            double confidence = (data['confidence'] as num?)?.toDouble() ?? 0.0;
            bool isLeaf = (data['is_leaf'] as bool?) ?? (confidence >= 0.50 && label != 'Not a Cabbage Leaf' && label != 'Not cabbage' && label != 'Image Not Recognized as Cabbage');

            if (confidence < 0.50 || !isLeaf || label == 'Not a Cabbage Leaf' || label == 'Not cabbage' || label == 'Image Not Recognized as Cabbage') {
              return {
                'label': 'Image Not Recognized as Cabbage',
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
          } else {
            debugPrint('TFLite Web: Render API returned status ${apiResponse.statusCode}, retrying...');
          }
        } catch (attemptErr) {
          debugPrint('TFLite Web: Attempt $attempts error: $attemptErr, retrying...');
        }

        // Brief delay before next retry attempt
        await Future.delayed(const Duration(milliseconds: 2000));
      }

      debugPrint('TFLite Web Error: Max retries reached.');
      return null;
    } catch (e) {
      debugPrint('TFLite Web Error: $e');
      return null;
    }
  }

  @override
  void dispose() {}
}
