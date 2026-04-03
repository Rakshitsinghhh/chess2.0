declare module "onnxruntime-node" {
  export class Tensor {
    constructor(type: string, data: Float32Array | BigInt64Array | Uint8Array, dims: readonly number[]);
    readonly data: Float32Array | BigInt64Array | Uint8Array;
    readonly dims: readonly number[];
  }

  export interface InferenceSession {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }

  export const InferenceSession: {
    create(
      pathOrBuffer: string | ArrayBuffer | Uint8Array,
      options?: { executionProviders?: string[] }
    ): Promise<InferenceSession>;
  };
}
