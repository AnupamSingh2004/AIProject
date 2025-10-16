import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-sans'
})

export const metadata: Metadata = {
  title: "StyleAI - Your AI Fashion Companion",
  description: "Get personalized fashion recommendations, wardrobe management, and style analysis powered by AI",
  keywords: ['fashion', 'AI', 'style', 'wardrobe', 'recommendations', 'outfit', 'color analysis'],
  authors: [{ name: 'StyleAI Team' }],
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <body className="min-h-screen flex flex-col antialiased light">
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                const theme = localStorage.getItem('fashion-theme') || 'light';
                document.documentElement.className = theme + ' ' + document.documentElement.className;
              })();
            `,
          }}
        />
        <ThemeProvider defaultTheme="light" storageKey="fashion-theme">
          <Header />
          <main className="flex-1">
            {children}
          </main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}

