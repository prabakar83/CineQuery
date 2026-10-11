import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CineQuery",
  description: "Upload a movie, then ask anything about it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="container">
            <Link href="/" className="logo">Cine<span>Query</span></Link>
            <span className="muted small">ask anything about your videos</span>
          </div>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
