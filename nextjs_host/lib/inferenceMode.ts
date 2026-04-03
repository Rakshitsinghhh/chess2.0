/**
 * USE_ONNX=1 or true → run model with onnxruntime-node + public/model.onnx
 * Otherwise → proxy to Python MODEL_SERVER_URL (default localhost:8001)
 */
export function shouldUseOnnx(): boolean {
  const v = (process.env.USE_ONNX ?? "").toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}
