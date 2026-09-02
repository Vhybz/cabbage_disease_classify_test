abstract class TFLiteServiceInterface {
  Future<void> loadModel();
  Future<Map<String, dynamic>?> classifyImage(dynamic imageInput);
  void dispose();
}
