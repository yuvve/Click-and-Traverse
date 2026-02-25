import onnxruntime as rt


class ONNXPolicy:
    def __init__(self, onnx_model_path: str, output_names=None, providers=None):
        self.onnx_model_path = onnx_model_path
        self.providers = ["CPUExecutionProvider"] if providers is None else providers
        self.output_names = ["continuous_actions"] if output_names is None else output_names
        self._load_model()

    def _load_model(self):
        self.policy = rt.InferenceSession(self.onnx_model_path, providers=self.providers)

    def inference(self, input_data):
        actions = self.policy.run(self.output_names, input_data)[0]
        return actions
