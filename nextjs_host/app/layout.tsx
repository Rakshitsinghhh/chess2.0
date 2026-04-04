import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Chess2.0 - Play vs Neural Network",
  description: "Play chess against your PyTorch model through Next.js",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>{children}</body>
    </html>
  );
}