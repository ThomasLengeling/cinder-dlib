#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "ciDlib.h"

using namespace ci;
using namespace ci::app;
using namespace std;


class ClasificationApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
    
    dml::ofxLearnSVM classifier;
    vector<vector<double> > trainingExamples;
    vector<int> trainingLabels;
};

void ClasificationApp::setup()
{
    
    // add 5000 samples to training set
    for (int i=0; i<5000; i++)
    {
        // our samples have two features: x, and y,
        // which are bound between (0, 1).
        // note: your feature values don't need to be between 0, 1
        // but best practice is to pre-normalize because it's faster
        // and ensures parity of feature influences
        
        vector<double> sample;
        sample.push_back(ci::randFloat(1));
        sample.push_back(ci::randFloat(1));
        
        // our label contains 3 possible classes, which roughly
        // correspond to the distance from the center of the screen
        // with some noise thrown in
        int label;
        float distFromCenter = distance(ci::vec2(sample[0], sample[1]), ci::vec2(0.5, 0.5));
        if (distFromCenter < ci::randFloat(0.1, 0.25)) {
            label = 1;
        }
        else if (distFromCenter < ci::randFloat(0.15, 0.45)) {
            label = 2;
        }
        else {
            label = 3;
        }
        
        // save our samples
        trainingExamples.push_back(sample);
        trainingLabels.push_back(label);
        
        // add sample to our classifier
        classifier.addTrainingInstance(sample, label);
    }
    
    classifier.train();
}

void ClasificationApp::mouseDown( MouseEvent event )
{
}

void ClasificationApp::update()
{
}

void ClasificationApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
    

    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        int trainingLabel = trainingLabels[i];
        if (trainingLabel == 1) {
            gl::color(255, 0, 0);
        }
        else if (trainingLabel == 2) {
             gl::color(0, 255, 0);
        }
        else if (trainingLabel == 3) {
             gl::color(0, 0, 255);
        }
        gl::drawSolidCircle(ci::vec2(trainingExample[0] * getWindowWidth(), trainingExample[1] * getWindowHeight()), 5);
    }
    
    /*
    // classify a new sample
    vector<double> sample;
    sample.push_back((double)ofGetMouseX()/getWindowWidth());
    sample.push_back((double)ofGetMouseY()/getWindowHeight());
    
    int label = classifier.predict(sample);
    
    if (label == 1) {
        ofSetColor(255, 0, 0);
    }
    else if (label == 2) {
        ofSetColor(0, 255, 0);
    }
    else if (label == 3) {
        ofSetColor(0, 0, 255);
    }
    ofCircle(ofGetMouseX(), ofGetMouseY(), ofMap(sin(0.1*ofGetFrameNum()), -1, 1, 5, 35));
     */
}

CINDER_APP( ClasificationApp, RendererGl )
