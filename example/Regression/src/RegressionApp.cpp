#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Rand.h"

#include "ciDlib.h"


using namespace ci;
using namespace ci::app;
using namespace std;

class RegressionApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
    
    
    dml::ofxLearnMLP mlp;
    dml::ofxLearnSVR svr;
    
    ci::vec2 mMousePos;
    
    vector<vector<double> > trainingExamples;
    vector<double> trainingLabels;
};

void RegressionApp::setup()
{
    
    // create a noisy training set
    // objective is to predict y as function of x
    for (int i=0; i<300; i++)
    {
        double x = ci::randFloat(getWindowWidth());
        double y = 0.00074 * pow(x, 2) + 0.0095*x + ci::randFloat(-80, 80);
        
        // in this example, we bound all input and output variables to (0,1).
        // this isn't strictly required but is best practice because it ensures
        // features are at parity in influence, and training is generally faster.
        // for MLP normalization to (0, 1) is required.
        x = glm::clamp<float>(lmap<float>(x, 0, getWindowWidth(), 0, 1), 0, 1);
        y = glm::clamp<float>(lmap<float>(y, 0, getWindowHeight(), 0, 1), 0, 1);
        
        // for this example, each instance contains one feature.
        // in general, an instance vector can contain any number
        // of elements, but must stay fixed for a single classifier/regressor
        vector<double> sample;
        sample.push_back(x);
        
        mlp.addTrainingInstance(sample, y);
        svr.addTrainingInstance(sample, y);
        
        trainingExamples.push_back(sample);
        trainingLabels.push_back(y);
    }
    
    // we have two different algorithms for doing regression:
    // MLP = multilayer perceptron (neural network)
    // SVR = support vector regression
    
    mlp.train();
    svr.train();
    
}

void RegressionApp::mouseDown( MouseEvent event )
{
    mMousePos = event.getPos();
}

void RegressionApp::update()
{
}

void RegressionApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
    
    gl::color(1, 0, 0, 0.6);
    for (int i=0; i<trainingExamples.size(); i++) {
        vector<double> trainingExample = trainingExamples[i];
        gl::drawSolidCircle(ci::vec2(getWindowWidth() * trainingExample[0], getWindowHeight() * trainingLabels[i]), 5);
    }
    
    // predict regression for mouseX
    float x = (float) mMousePos.x / getWindowWidth();
    
    vector<double> sample;
    sample.push_back(x);
    
    double mlpPrediction = mlp.predict(sample);
    
    gl::color(0, 1, 0, 0.5);
    gl::drawSolidCircle(ci::vec2(getWindowWidth() * x, getWindowHeight() * mlpPrediction), 20);
    //ofSetColor(0);
    //ofDrawBitmapString("MLP", ofGetWidth() * x, ofGetHeight() * mlpPrediction);
    
    double svrPrediction = svr.predict(sample);
    
    gl::color(0, 0, 1, 0.5);
    gl::drawSolidCircle(ci::vec2(getWindowWidth() * x, getWindowHeight() * svrPrediction), 20);
}

CINDER_APP( RegressionApp, RendererGl )
