import { Button } from "@/components/ui/button";

interface NavigationProps {
  showAbout: boolean;
  onHomeClick: () => void;
  onAboutClick: () => void;
}

export function Navigation({ showAbout, onHomeClick, onAboutClick }: NavigationProps) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-lg">C</span>
            </div>
            <span className="text-2xl font-bold text-primary">
              CareerLink
            </span>
          </div>
          
          <div className="flex items-center space-x-4">
            <Button
              variant={!showAbout ? "hero" : "ghost"}
              size="sm"
              onClick={onHomeClick}
            >
              Home
            </Button>
            <Button
              variant={showAbout ? "hero" : "ghost"}
              size="sm"
              onClick={onAboutClick}
            >
              About
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}