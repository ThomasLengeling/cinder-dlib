//
//  ciDlib.hpp
//  Clasification
//
//  Created by tom on 1/3/16.
//
//

#pragma once

#include "cinder/Log.h"
#include "cinder/Rand.h"


#include "dlib/svm.h"
#include "dlib/mlp.h"
#include "dlib/svm/svm_threaded.h"
#include "dlib/matrix/matrix_abstract.h"
#include "dlib/statistics/statistics.h"
#include "dlib/svm/one_vs_one_trainer.h"

#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>


//machine learning library dlib
namespace dml {
    
    // sample type
    typedef dlib::matrix<double, 0, 1>                  sample_type;
    
    // kernel types
    typedef dlib::radial_basis_kernel<sample_type>      rbf_kernel_type;
    typedef dlib::polynomial_kernel<sample_type>        poly_kernel_type;
    
    // trainer types
    typedef dlib::any_trainer<sample_type>              any_trainer;
    typedef dlib::one_vs_one_trainer<any_trainer>       ovo_trainer;
    typedef dlib::mlp::kernel_1a_c                      mlp_trainer_type;
    
    class ofxLearn
    {
    public:
        ofxLearn() { }
        virtual ~ofxLearn() { }
        void svd();
        
        virtual void train() { }
        
        virtual void saveModel(std::string path) { }
        virtual void loadModel(std::string path) { }
        
        inline sample_type vectorToSample(std::vector<double> sample_);
    };
    
    
    
    class ofxLearnSupervised : public ofxLearn
    {
    public:
        ofxLearnSupervised() : ofxLearn() {}
        
        void addTrainingInstance(std::vector<double> sample, double label);
        void addSample(sample_type sample, double label);
        void clearTrainingInstances();
        
        virtual double predict(std::vector<double> & sample) { }
        virtual double predict(sample_type & sample) { }
        
    protected:
        
        std::vector<sample_type> samples;
        std::vector<double> labels;
        
    };
    
    class ofxLearnUnsupervised : public ofxLearn
    {
    public:
        ofxLearnUnsupervised() : ofxLearn() {}
        
        void addTrainingInstance(std::vector<double> sample);
        void addSample(sample_type sample);
        void clearTrainingInstances();
        
    protected:
        
        std::vector<sample_type> samples;
        
    };
    
    
    class ofxLearnMLP : public ofxLearnSupervised
    {
    public:
        ofxLearnMLP();
        ~ofxLearnMLP();
        
        void train();
        double predict(std::vector<double> & sample);
        double predict(sample_type & sample);
        
        void setHiddenLayers(int hiddenLayers) {this->hiddenLayers = hiddenLayers;}
        void setTargetRmse(float targetRmse) {this->targetRmse = targetRmse;}
        void setMaxSamples(int maxSamples) {this->maxSamples = maxSamples;}
        
        int getHiddenLayers() {return hiddenLayers;}
        float getTargetRmse() {return targetRmse;}
        int getMaxSamples() {return maxSamples;}
        
        mlp_trainer_type * getTrainer() {return mlp_trainer;}
        
    private:
        
        mlp_trainer_type *mlp_trainer;
        
        int hiddenLayers;
        float targetRmse;
        int maxSamples;
    };
    
    
    
    class ofxLearnSVR : public ofxLearnSupervised
    {
    public:
        ofxLearnSVR();
        ~ofxLearnSVR();
        
        void train();
        void trainWithGridParameterSearch();
        double predict(std::vector<double> & sample);
        double predict(sample_type & sample);
        
    private:
        
        dlib::svr_trainer<rbf_kernel_type> trainer;
        dlib::decision_function<rbf_kernel_type> df;
    };
    
    
    class ofxLearnSVM : public ofxLearnSupervised
    {
    public:
        ofxLearnSVM();
        ~ofxLearnSVM();
        
        void train();
        void trainWithGridParameterSearch();
        
        double predict(std::vector<double> & sample);
        double predict(sample_type & sample);
        
        void saveModel(std::string path)
        {
            const char *filepath = path.c_str();
            std::ofstream fout(filepath, std::ios::binary);
            dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel_type> > df2, df3;
            df2 = df;
            serialize(df2, fout);
            
        }
        
        void loadModel(std::string path) {
            const char *filepath = path.c_str();
            std::ifstream fin(filepath, std::ios::binary);
            dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel_type> > df2;
            dlib::deserialize(df2, fin);
            df = df2;
        }
        
    private:
        
        ovo_trainer trainer;
        
        //dlib::svm_nu_trainer<poly_kernel_type> poly_trainer;
        dlib::krr_trainer<rbf_kernel_type> rbf_trainer;
        dlib::one_vs_one_decision_function<ovo_trainer> df;
    };
    
    
    class ofxLearnKMeans : public ofxLearnUnsupervised
    {
    public:
        ofxLearnKMeans();
        
        int getNumClusters() {return numClusters;}
        void setNumClusters(int numClusters);
        void train();
        std::vector<int> & getClusters() {return clusters;}
        
    private:
        std::vector<int> clusters;
        int numClusters;
    };
    
    

    ////////////////////////
    
    class ofxLearnThreaded : public ofxLearn
    {
    public:
        ofxLearnThreaded();
        ~ofxLearnThreaded();
        
        void beginTraining();
        
        bool  isThreadRunning();
        
        template <typename L, typename M>
        void beginTraining(L *listener, M method);
        
        bool isTrained() {return mTraning;}
        
    private:
        std::shared_ptr<std::thread>		mThread;
        std::mutex                          mMutex;
        bool                                mTraning;
        
        void threadedFunction();
        virtual void threadedTrainer() {};
    
    };
    
    template <typename L, typename M>
    void ofxLearnThreaded::beginTraining(L *listener, M method)
    {
        mThread = std::shared_ptr<std::thread>( new std::thread( std::bind( &ofxLearnThreaded::threadedFunction, this)));
        beginTraining();
    }
    
    class ofxLearnMLPThreaded : public ofxLearnMLP, public ofxLearnThreaded {
        void threadedTrainer() {ofxLearnMLP::train();}
    };
    
    class ofxLearnSVRThreaded : public ofxLearnSVR, public ofxLearnThreaded {
        void threadedTrainer() {ofxLearnSVR::train();}
    };
    
    class ofxLearnSVMThreaded : public ofxLearnSVM, public ofxLearnThreaded {
        void threadedTrainer() {ofxLearnSVM::train();}
    };
    
    class ofxLearnKMeansThreaded : public ofxLearnKMeans, public ofxLearnThreaded {
        void threadedTrainer() {ofxLearnKMeans::train();}
    };
}