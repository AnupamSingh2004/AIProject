'use client';

import { useState, useRef } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';

export default function AnalyzePage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    
    setAnalyzing(true);
    
    // Simulate AI analysis (replace with actual API call)
    setTimeout(() => {
      setAnalysisResult({
        skinType: 'Type IV (Medium-Dark)',
        undertone: 'Warm',
        dominantColor: '#D4A57A',
        confidence: 92,
        recommendations: [
          'Earth tones (browns, terracotta, olive)',
          'Warm jewel tones (amber, coral, bronze)',
          'Warm neutrals (camel, cream, warm gray)',
          'Avoid: Cool blues and cool grays',
        ],
      });
      setAnalyzing(false);
    }, 2000);
  };

  const clearImage = () => {
    setSelectedImage(null);
    setAnalysisResult(null);
    setSaveMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSaveToProfile = async () => {
    if (!analysisResult) return;

    setSaving(true);
    setSaveMessage(null);

    try {
      // Get or create userId - use a proper UUID format
      let userId = localStorage.getItem('userId');
      if (!userId) {
        // Generate a valid UUID v4
        userId = crypto.randomUUID();
        localStorage.setItem('userId', userId);
      }

      const response = await fetch('/api/skin-tone/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          skinType: analysisResult.skinType,
          undertone: analysisResult.undertone,
          dominantColor: analysisResult.dominantColor,
          confidence: analysisResult.confidence,
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        console.error('API Error:', data);
        throw new Error(data.error || 'Failed to save skin tone analysis');
      }

      setSaveMessage('✓ Saved to your profile successfully!');
      
      // Redirect to recommendations after 2 seconds
      setTimeout(() => {
        router.push('/recommendations');
      }, 2000);
    } catch (error) {
      console.error('Error saving skin tone analysis:', error);
      setSaveMessage(`✗ Failed to save: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen section-alt py-20">
      <div className="container-wide">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-responsive-md mb-4 text-foreground font-bold">Analyze Your Skin Tone</h1>
          <p className="text-responsive-xs text-muted max-w-2xl mx-auto leading-relaxed">
            Upload a clear photo of your face and let our AI analyze your skin tone for personalized recommendations
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-12">
          {/* Upload Section */}
          <div className="card p-6 animate-fade-in">
            <h2 className="text-responsive-sm mb-6 text-foreground font-semibold">Upload Your Photo</h2>
            
            {/* Guidelines */}
            <div className="mb-6 p-4 fashion-glow rounded-lg">
              <h3 className="font-medium text-foreground mb-3 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-accent" />
                <span className="text-sm sm:text-base">Guidelines for Best Results</span>
              </h3>
              <ul className="text-muted space-y-1 text-xs sm:text-sm ml-6">
                <li>• Use natural lighting</li>
                <li>• Face clearly visible</li>
                <li>• No filters or heavy makeup</li>
                <li>• Look directly at camera</li>
                <li>• Neutral expression</li>
              </ul>
            </div>

            {/* Upload Area */}
            {!selectedImage ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-primary-300 rounded-2xl p-12 text-center cursor-pointer hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-all group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <div className="flex flex-col items-center gap-6">
                  <div className="w-20 h-20 rounded-2xl fashion-gradient flex items-center justify-center shadow-theme-md group-hover:scale-110 transition-transform">
                    <Upload className="h-10 w-10 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-foreground mb-2 text-lg">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-muted">
                      PNG, JPG, HEIC up to 10MB
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Image Preview */}
                <div className="relative rounded-xl overflow-hidden">
                  <img
                    src={selectedImage}
                    alt="Preview"
                    className="w-full h-80 object-cover"
                  />
                  <button
                    onClick={clearImage}
                    className="btn btn-ghost btn-sm absolute top-4 right-4"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                {/* Analyze Button */}
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="btn btn-primary btn-lg w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="h-6 w-6 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="h-6 w-6" />
                      <span>Analyze Skin Tone</span>
                    </>
                  )}
                </button>

                <button
                  onClick={clearImage}
                  className="btn btn-outline btn-lg w-full"
                >
                  Upload Different Photo
                </button>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="outfit-card animate-fade-in">
            <h2 className="heading-secondary mb-8 text-foreground">Analysis Results</h2>

            {!analysisResult ? (
              <div className="flex flex-col items-center justify-center h-80 text-center">
                <div className="w-24 h-24 rounded-2xl bg-surface flex items-center justify-center mb-6 shadow-theme-sm">
                  <Camera className="h-12 w-12 text-muted" />
                </div>
                <p className="text-muted text-lg">
                  Upload a photo and click analyze to see your results
                </p>
              </div>
            ) : (
              <div className="space-y-8">
                {/* Confidence Score */}
                <div className="p-6 bg-accent-50 dark:bg-accent-900/20 rounded-xl border border-accent-200 dark:border-accent-700">
                  <div className="flex items-center gap-3 mb-3">
                    <CheckCircle2 className="h-6 w-6 text-accent" />
                    <span className="font-semibold text-foreground text-lg">
                      Analysis Complete
                    </span>
                  </div>
                  <p className="text-muted">
                    Confidence: {analysisResult.confidence}%
                  </p>
                </div>

                {/* Skin Type */}
                <div>
                  <h3 className="font-semibold text-foreground mb-3 text-lg">
                    Skin Type (Fitzpatrick Scale)
                  </h3>
                  <p className="text-xl text-primary-600 font-medium">
                    {analysisResult.skinType}
                  </p>
                </div>

                {/* Undertone */}
                <div>
                  <h3 className="font-semibold text-foreground mb-3 text-lg">Undertone</h3>
                  <p className="text-xl text-secondary-600 font-medium">
                    {analysisResult.undertone}
                  </p>
                </div>

                {/* Dominant Color */}
                <div>
                  <h3 className="font-semibold text-foreground mb-4 text-lg">
                    Dominant Skin Color
                  </h3>
                  <div className="flex items-center gap-4">
                    <div
                      className="w-20 h-20 rounded-xl border-2 border-border shadow-theme-md"
                      style={{ backgroundColor: analysisResult.dominantColor }}
                    ></div>
                    <span className="font-mono text-lg text-muted">
                      {analysisResult.dominantColor}
                    </span>
                  </div>
                </div>

                {/* Recommendations */}
                <div>
                  <h3 className="font-semibold text-foreground mb-4 text-lg">
                    Color Recommendations
                  </h3>
                  <ul className="space-y-3">
                    {analysisResult.recommendations.map((rec: string, index: number) => (
                      <li
                        key={index}
                        className="flex items-start gap-3 text-foreground"
                      >
                        <CheckCircle2 className="h-6 w-6 text-primary-600 flex-shrink-0 mt-0.5" />
                        <span className="text-lg">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Action Buttons */}
                <div className="pt-6 space-y-4">
                  <button 
                    onClick={() => router.push('/recommendations')}
                    className="btn btn-primary btn-lg w-full"
                  >
                    View Recommended Outfits
                  </button>
                  <button 
                    onClick={handleSaveToProfile}
                    disabled={saving}
                    className="btn btn-secondary btn-lg w-full flex items-center justify-center gap-2"
                  >
                    {saving ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      'Save to Profile'
                    )}
                  </button>
                  {saveMessage && (
                    <p className={`text-center text-sm ${
                      saveMessage.startsWith('✓') ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {saveMessage}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-16 grid-responsive md:grid-cols-3">
          <div className="outfit-card group">
            <h3 className="font-bold text-xl mb-4 text-foreground">Privacy First</h3>
            <p className="text-muted leading-relaxed">
              Your photos are analyzed securely and never shared or stored without your permission.
            </p>
          </div>
          <div className="outfit-card group">
            <h3 className="font-bold text-xl mb-4 text-foreground">AI-Powered</h3>
            <p className="text-muted leading-relaxed">
              Our advanced AI uses computer vision and color theory to provide accurate skin tone analysis.
            </p>
          </div>
          <div className="outfit-card group">
            <h3 className="font-bold text-xl mb-4 text-foreground">Personalized</h3>
            <p className="text-muted leading-relaxed">
              Get recommendations tailored specifically to your unique skin tone and undertone.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

