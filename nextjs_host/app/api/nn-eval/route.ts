import { NextRequest, NextResponse } from "next/server";
import { shouldUseOnnx } from "@/lib/inferenceMode";
import { onnxEvalValue } from "@/lib/onnx/fastPredict";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const fen = body?.fen as string | undefined;
    if (!fen || typeof fen !== "string") {
      return NextResponse.json({ error: "Missing or invalid `fen`" }, { status: 400 });
    }

    if (shouldUseOnnx()) {
      try {
        const data = await onnxEvalValue(fen);
        return NextResponse.json(data);
      } catch (err) {
        return NextResponse.json(
          { error: "ONNX eval failed", details: String(err) },
          { status: 500 }
        );
      }
    }

    const backendUrl = process.env.MODEL_SERVER_URL || "http://127.0.0.1:8001";

    let res: Response;
    try {
      res = await fetch(`${backendUrl}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen }),
        cache: "no-store",
        // no AbortController, no timeout — wait as long as it takes
      });
    } catch (err) {
      return NextResponse.json(
        { error: "Failed to reach model server", details: String(err) },
        { status: 503 }
      );
    }

    const data = await res.json();
    if (!res.ok) {
      return NextResponse.json(
        { error: data?.detail || data?.error || "Eval server error", details: data },
        { status: 502 }
      );
    }
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to evaluate position", details: String(err) },
      { status: 500 }
    );
  }
}