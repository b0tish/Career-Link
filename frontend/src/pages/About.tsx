import React from 'react';
import { Card } from '@/components/ui/card';
import { Star, TrendingUp } from 'lucide-react';
import { Navigation } from '@/components/Navigation';

const About = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />
      <main className="pt-20"> {/* Add padding to main content */}
        <section className="py-32 px-6">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-16">
              <h1 className="text-5xl font-bold mb-8 text-primary">
                About CareerLink
              </h1>
            </div>

            <Card className="p-12   border border-white/20 ">
              <div className="space-y-8">
                <p className="text-lg text-muted-foreground leading-relaxed">
                  CareerLink is a revolutionary career platform designed to
                  bridge the gap between talented professionals and exciting
                  job opportunities. We leverage cutting-edge AI technology to
                  provide personalized career solutions for job seekers at
                  every stage of their journey.
                </p>

                <p className="text-lg text-muted-foreground leading-relaxed">
                  From intelligent resume analysis to professional networking
                  opportunities, CareerLink empowers users to manage their
                  career journey with confidence and achieve their
                  professional goals.
                </p>

                <div className="grid md:grid-cols-2 gap-8 mt-12">
                  <div className="p-8   rounded-2xl border border-white/20">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
                        <Star className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-2xl font-bold">Our Mission</h3>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      To democratize access to career opportunities and
                      connect talent with opportunity globally, regardless of
                      background or location.
                    </p>
                  </div>

                  <div className="p-8   rounded-2xl border border-white/20">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-2xl font-bold">Our Vision</h3>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      Creating a future where everyone has access to
                      meaningful career opportunities and the tools to achieve
                      their professional aspirations.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
};

export default About;
