import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Chess2.0 - Play vs Neural Network",
  description: "Play chess against your PyTorch model through Next.js"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

