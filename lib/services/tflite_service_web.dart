import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'tflite_service_interface.dart';

class TFLiteService implements TFLiteServiceInterface {
  final List<String> _labels = [
    'Alternaria Leaf Spot',
    'Bacterial Soft Rot',
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
  Future<Map<String, dynamic>?> classifyImage(String imageSource) async {
    try {
      debugPrint('TFLite Web: Sending image bytes to Render API...');
      
      // 1. Fetch image bytes directly from the URL provided (blob or network)
      final response = await http.get(Uri.parse(imageSource));
      final Uint8List imageBytes = response.bodyBytes;

      // 2. Call Render API
      final String apiUrl = dotenv.env['RENDER_API_URL'] ?? 'https://cabbage-disease-classify-test.onrender.com/predict';
      
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: 'image.jpg',
      ));

      final streamedResponse = await request.send();
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
      } else {
        debugPrint('TFLite Web: Render API Error: ${apiResponse.statusCode} - ${apiResponse.body}');
        return null;
      }
    } catch (e) {
      debugPrint('TFLite Web Error: $e');
      return null;
    }
  }

  @override
  void dispose() {}
}
