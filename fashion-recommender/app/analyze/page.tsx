'use client';

import { useState, useRef } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import Image from 'next/image';

export default function AnalyzePage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 transition-colors duration-300">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl font-bold mb-4 text-gray-900">Analyze Your Skin Tone</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a clear photo of your face and let our AI analyze your skin tone for personalized recommendations
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-2xl shadow-lg p-8 animate-fade-in border border-gray-100 transition-colors duration-300">
            <h2 className="text-2xl font-bold mb-6 text-gray-900">Upload Your Photo</h2>
            
            {/* Guidelines */}
            <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
              <h3 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                Guidelines for Best Results
              </h3>
              <ul className="text-sm text-blue-800 space-y-1 ml-7">
                <li>• Use natural lighting (near a window is ideal)</li>
                <li>• Face should be clearly visible</li>
                <li>• No filters or heavy makeup</li>
                <li>• Look directly at the camera</li>
                <li>• Neutral expression works best</li>
              </ul>
            </div>

            {/* Upload Area */}
            {!selectedImage ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-purple-300 rounded-xl p-12 text-center cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-all"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-purple-100 flex items-center justify-center">
                    <Upload className="h-8 w-8 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900 mb-1">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-gray-600">
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
                    className="absolute top-4 right-4 p-2 bg-white rounded-full shadow-lg hover:bg-gray-100 transition-colors"
                  >
                    <X className="h-5 w-5 text-gray-700" />
                  </button>
                </div>

                {/* Analyze Button */}
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="w-full py-4 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="h-5 w-5" />
                      <span>Analyze Skin Tone</span>
                    </>
                  )}
                </button>

                <button
                  onClick={clearImage}
                  className="w-full py-3 bg-gray-100 text-gray-700 rounded-xl font-medium hover:bg-gray-200 transition-colors"
                >
                  Upload Different Photo
                </button>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-2xl shadow-lg p-8 animate-fade-in border border-gray-100 transition-colors duration-300">
            <h2 className="text-2xl font-bold mb-6 text-gray-900">Analysis Results</h2>

            {!analysisResult ? (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                  <Camera className="h-10 w-10 text-gray-400" />
                </div>
                <p className="text-gray-500">
                  Upload a photo and click analyze to see your results
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Confidence Score */}
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                    <span className="font-semibold text-green-900">
                      Analysis Complete
                    </span>
                  </div>
                  <p className="text-sm text-green-800">
                    Confidence: {analysisResult.confidence}%
                  </p>
                </div>

                {/* Skin Type */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">
                    Skin Type (Fitzpatrick Scale)
                  </h3>
                  <p className="text-lg text-purple-600 font-medium">
                    {analysisResult.skinType}
                  </p>
                </div>

                {/* Undertone */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Undertone</h3>
                  <p className="text-lg text-pink-600 font-medium">
                    {analysisResult.undertone}
                  </p>
                </div>

                {/* Dominant Color */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">
                    Dominant Skin Color
                  </h3>
                  <div className="flex items-center gap-3">
                    <div
                      className="w-16 h-16 rounded-lg border-2 border-gray-300 shadow-md"
                      style={{ backgroundColor: analysisResult.dominantColor }}
                    ></div>
                    <span className="font-mono text-sm text-gray-600">
                      {analysisResult.dominantColor}
                    </span>
                  </div>
                </div>

                {/* Recommendations */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">
                    Color Recommendations
                  </h3>
                  <ul className="space-y-2">
                    {analysisResult.recommendations.map((rec: string, index: number) => (
                      <li
                        key={index}
                        className="flex items-start gap-2 text-gray-700"
                      >
                        <CheckCircle2 className="h-5 w-5 text-purple-600 flex-shrink-0 mt-0.5" />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Action Buttons */}
                <div className="pt-4 space-y-3">
                  <button className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors">
                    View Recommended Outfits
                  </button>
                  <button className="w-full py-3 bg-pink-600 text-white rounded-xl font-semibold hover:bg-pink-700 transition-colors">
                    Save to Profile
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-xl p-6 shadow-md border border-gray-100 transition-colors duration-300">
            <h3 className="font-bold text-lg mb-2 text-gray-900">Privacy First</h3>
            <p className="text-gray-600 text-sm">
              Your photos are analyzed securely and never shared or stored without your permission.
            </p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-md border border-gray-100 transition-colors duration-300">
            <h3 className="font-bold text-lg mb-2 text-gray-900">AI-Powered</h3>
            <p className="text-gray-600 text-sm">
              Our advanced AI uses computer vision and color theory to provide accurate skin tone analysis.
            </p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-md border border-gray-100 transition-colors duration-300">
            <h3 className="font-bold text-lg mb-2 text-gray-900">Personalized</h3>
            <p className="text-gray-600 text-sm">
              Get recommendations tailored specifically to your unique skin tone and undertone.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

