#include <iostream>
#include <sstream>

#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include <map>

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, int> Prediction;
int N = 5;
std::vector<std::string> labels_;

static bool PairCompare(const std::pair<int, int> &lhs,
                        const std::pair<int, int> &rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<int> &v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

int main(int argc, char **argv) {
    FLAGS_alsologtostderr = 1;
    std::cout << "Scoring started!" << std::endl;
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL_PROTOTXT WEIGHTS_CAFFEMODEL LABEL_FILE [ITERATIONS]" << std::endl;
        return 1;
    }
    int iterations = 1;
    if (argc == 5) {
        std::istringstream ss(argv[4]);
        int x;
        if (!(ss >> x)) {
            std::cerr << "Invalid number iterations: " << argv[4] << '\n';
        }
        else {
            iterations = x;
        }
    }
    const std::string param_file = argv[1];
    const std::string trained_filename = argv[2];
    const std::string label_file = argv[3];

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    std::string line;
    while (std::getline(labels, line))
        labels_.push_back(std::string(line));

//    Caffe::set_mode(Caffe::CPU);
    caffe::Net<int> caffe_net(param_file, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(trained_filename);

    int loss = 0;
    for (int i = 0; i < iterations; ++i) {
        int iter_loss;
        const std::vector<caffe::Blob<int> *> &result =
                caffe_net.Forward(&iter_loss);
        loss += iter_loss;

        /* Copy the output layer to a std::vector */
        caffe::Blob<int> *output_layer = caffe_net.output_blobs()[1];

        const int *begin = output_layer->cpu_data();
        const int *end = begin + output_layer->channels();
        std::vector<int> output = std::vector<int>(begin, end);

        CHECK_EQ(labels_.size(), output_layer->channels())
            << "Number of labels is different from the output layer dimension.";

        N = std::min<int>(labels_.size(), N);
        std::vector<int> maxN = Argmax(output, N);
        std::vector<Prediction> predictions;
        for (int ii = 0; ii < N; ++ii) {
            int idx = maxN[ii];
            predictions.push_back(std::make_pair(labels_[idx], output[idx]));
        }

        /* Print the top N predictions. */
        std::cout << "Prediction: " << i << std::endl;
        for (size_t iii = 0; iii < predictions.size(); ++iii) {
            Prediction p = predictions[iii];
            std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
            << p.first << "\"" << std::endl;
        }
    }

    std::cout << "Scoring finished!" << std::endl;
    return 0;
}

