import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, FileText, CheckCircle } from "lucide-react";

interface UploadSectionProps {
  onFileUpload: (file: File) => void;
  onProcess: () => void;
  uploadedFile: File | null;
  isProcessing: boolean;
}

export function UploadSection({ onFileUpload, onProcess, uploadedFile, isProcessing }: UploadSectionProps) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <section id="upload-section" className="py-20 px-6">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
          Upload Your Documents
        </h2>
        <p className="text-xl text-muted-foreground mb-12">
          Upload your resume, cover letter, or career documents to get personalized job matches.
        </p>

        <Card className="p-8 bg-gradient-card backdrop-blur-sm border border-white/20 shadow-glass">
          <div
            className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ${
              dragActive
                ? "border-primary bg-primary/5 scale-105"
                : "border-muted hover:border-primary hover:bg-primary/5"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              onChange={handleFileChange}
              accept=".pdf,.doc,.docx,.txt"
            />
            
            <div className="flex flex-col items-center space-y-4">
              <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <div>
                <p className="text-lg font-semibold mb-2">
                  Click to upload or drag and drop
                </p>
                <p className="text-muted-foreground">
                  PDF, DOC, DOCX, TXT (max 10MB)
                </p>
              </div>
            </div>
          </div>

          {uploadedFile && (
            <div className="mt-8 p-6 bg-accent/10 rounded-2xl border border-accent/20">
              <div className="flex items-center justify-center space-x-3">
                <CheckCircle className="w-6 h-6 text-accent" />
                <div className="text-center">
                  <p className="font-semibold text-accent">File uploaded successfully!</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    <FileText className="w-4 h-4 inline mr-1" />
                    {uploadedFile.name} • {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            </div>
          )}

          {uploadedFile && (
            <div className="mt-8">
              <Button 
                variant="hero" 
                size="lg" 
                onClick={onProcess}
                disabled={isProcessing}
                className="min-w-48"
              >
                {isProcessing ? "Processing..." : "Find Matching Jobs"}
              </Button>
            </div>
          )}
        </Card>
      </div>
    </section>
  );
}