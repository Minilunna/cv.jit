/*
cv.jit.barcode

This file uses cv.jit.

cv.jit is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cv.jit is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with cv.jit.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
This file links to the OpenCV library <http://sourceforge.net/projects/opencvlibrary/>

Please refer to the  Intel License Agreement For Open Source Computer Vision Library.

Please also read the notes concerning technical issues with using the OpenCV library
in Jitter externals.
*/

#include "cvjit.h"
#include <opencv2/dnn/dnn.hpp>
#include <fstream>
#include <zxing/Binarizer.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/ReaderException.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/common/HybridBinarizer.h>
#include <exception>
#include <zxing/Exception.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
#include "zxing/MatSource.h"


using namespace c74::max;

constexpr cvjit::cstring default_classes = "classes.names";       // Classes name
constexpr cvjit::cstring default_config = "yolov3-tiny.cfg";      // Darknet configuration
constexpr cvjit::cstring default_weights = "yolov3-tiny.weights"; // Darknet Weights
constexpr float default_threshold = 0.6f;
constexpr long default_resize_frame_height = 416;
constexpr long default_resize_frame_width = 416;

typedef struct _cv_jit_barcode
{
	t_object ob;
	long ready{ 0 };
	long normalize{ 0 };

	float threshold{ 0.6f };

	cv::dnn::Net net;
	std::vector<std::string> class_names;
	std::vector<std::string> out_names;

	cv::Size resize_frame;

	void load_model(std::string classes, std::string config, std::string weights, float threshold, long height, long width)
	{
		ready = 0;
		this->threshold = threshold;
		this->resize_frame = cv::Size(height, width);

		// Load Model asynchronously
		std::string abs_classes_path = cvjit::get_absolute_path(classes);
		std::string abs_config_path = cvjit::get_absolute_path(config);
		std::string abs_weights_path = cvjit::get_absolute_path(weights);
		if (!abs_config_path.empty() || !abs_weights_path.empty())
		{
			object_post((t_object*)this, "Loading net from Darknet. This may take some time...");

			std::thread worker = std::thread(
				[this, abs_classes_path, abs_config_path, abs_weights_path]()
				{
					try
					{
						//Load Classes from file
						std::ifstream input(abs_classes_path);
						if (input.is_open())
						{
							std::string line;
							while (std::getline(input, line))
							{
								class_names.push_back(line);
							}
						}

						//Initialize neural network from darknet configuration files
						net = cv::dnn::readNetFromDarknet(abs_config_path, abs_weights_path);
						if (net.empty())
						{
							object_error((t_object*)this, "Could not load net...");
							return;
						}

						net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
						out_names = net.getUnconnectedOutLayersNames();

						std::string str = "";
						if (!out_names.empty())
						{
							str = "\noutput layers: \n";
							for (size_t i = 0; i != out_names.size(); ++i)
								str = str + "    " + out_names[i] + "\n";
						}
						object_post((t_object*)this,
							"Finished loading net from Darknet:\nConfig : %s \nWeight : %s \nClasses: %s \nDetection threshold : %9.6f\nResize : (%d,%d)%s",
							abs_config_path.c_str(), abs_weights_path.c_str(), abs_classes_path.c_str(), this->threshold, this->resize_frame.height, this->resize_frame.width, str.c_str()
						);


						ready = 1;
					}
					catch (cv::Exception& exception)
					{
						object_error((t_object*)this, "OpenCV error: %s", exception.what());
					}
				}
			);
			worker.detach();
		}
		else
		{
			object_error((t_object*)this, "Could not read net from Darknet: ( %s, %s )", abs_config_path.c_str(), abs_weights_path.c_str());
		}
	}

} t_cv_jit_barcode;

void* _cv_jit_barcode_class;

t_jit_err cv_jit_barcode_init(void);
t_cv_jit_barcode* cv_jit_barcode_new(void);
t_jit_err cv_jit_barcode_matrix_calc(t_cv_jit_barcode* x, void* inputs, void* outputs);
void cv_jit_barcode_read(t_cv_jit_barcode* x, t_symbol* s, short argc, t_atom* argv);
void cv_jit_barcode_free(t_cv_jit_barcode* x) {} // Nothing

t_jit_err cv_jit_barcode_init(void)
{
	t_jit_object* attr, * mop, * output;
	t_symbol* atsym;

	atsym = gensym("jit_attr_offset");

	_cv_jit_barcode_class = jit_class_new("cv_jit_barcode", (method)cv_jit_barcode_new, (method)cv_jit_barcode_free, sizeof(t_cv_jit_barcode), 0L);

	// add mop
	mop = (t_jit_object*)jit_object_new(_jit_sym_jit_mop, 1, 1);           // Object has one input and one output
	output = (t_jit_object*)jit_object_method(mop, _jit_sym_getoutput, 1); // Get a pointer to the output matrix

	jit_mop_single_type(mop, _jit_sym_char); // Set input type and planecount
	jit_mop_single_planecount(mop, 1);

	jit_mop_output_nolink(mop, 1); // Turn off output linking so that output matrix does not adapt to input

	jit_attr_setlong(output, _jit_sym_minplanecount, 4); // Four planes, for representing rectangle corners
	jit_attr_setlong(output, _jit_sym_maxplanecount, 4);
	jit_attr_setlong(output, _jit_sym_mindim, 1); // Only one dimension
	jit_attr_setlong(output, _jit_sym_maxdim, 1);
	jit_attr_setsym(output, _jit_sym_types, _jit_sym_float32); // Coordinates are returned with sub-pixel accuracy

	jit_class_addadornment(_cv_jit_barcode_class, mop);

	// add methods
	jit_class_addmethod(_cv_jit_barcode_class, (method)cv_jit_barcode_matrix_calc, "matrix_calc", A_CANT, 0L);
	jit_class_addmethod(_cv_jit_barcode_class, (method)cv_jit_barcode_read, "read", A_GIMME, 0L);

	// add attributes
	attr = (t_jit_object*)jit_object_new(_jit_sym_jit_attr_offset, "ready", _jit_sym_long, cvjit::Flags::private_set, (method)0L, (method)0L, calcoffset(t_cv_jit_barcode, ready));
	jit_class_addattr(_cv_jit_barcode_class, attr);

	// Normalize attribute
	jit_class_addattr(_cv_jit_barcode_class, cvjit::normalize_attr<t_cv_jit_barcode>());

	jit_class_register(_cv_jit_barcode_class);
	return JIT_ERR_NONE;
}

t_cv_jit_barcode* cv_jit_barcode_new(void)
{
	t_cv_jit_barcode* x = (t_cv_jit_barcode*)jit_object_alloc(_cv_jit_barcode_class);
	if (x)
	{
		x->load_model(default_classes, default_config, default_weights, default_threshold, default_resize_frame_height, default_resize_frame_width);
	}
	return x;
}

void cv_jit_barcode_read(t_cv_jit_barcode* x, t_symbol* s, short argc, t_atom* argv)
{
	if (x)
	{
		std::string classes = (argc > 0 && argv[0].a_type == A_SYM) ? argv[0].a_w.w_sym->s_name : default_classes;
		std::string config = (argc > 1 && argv[1].a_type == A_SYM) ? argv[1].a_w.w_sym->s_name : default_config;
		std::string weights = (argc > 2 && argv[2].a_type == A_SYM) ? argv[2].a_w.w_sym->s_name : default_weights;
		float threshold = (argc > 3 && argv[3].a_type == A_FLOAT) ? atom_getfloat(&argv[3]) : default_threshold;
		long height = (argc > 4 && argv[4].a_type == A_LONG) ? atom_getlong(&argv[4]) : default_resize_frame_height;
		long width = (argc > 5 && argv[5].a_type == A_LONG) ? atom_getlong(&argv[5]) : default_resize_frame_width;

		x->load_model(classes, config, weights, threshold, height, width);
	}
}

t_jit_err cv_jit_barcode_matrix_calc(t_cv_jit_barcode* x, void* inputs, void* outputs)
{
	std::vector<cv::Rect> faces;

	// Get pointers to matrix
	t_object* input_image_matrix = (t_object*)jit_object_method(inputs, _jit_sym_getindex, 0);

	t_object* out_matrix = (t_object*)jit_object_method(outputs, _jit_sym_getindex, 0);

	if (x && input_image_matrix && out_matrix && x->ready)
	{
		// Lock the matrices
		cvjit::Savelock savelocks[] = { input_image_matrix, out_matrix };

		// Wrap the matrices
		cvjit::JitterMatrix input_image(input_image_matrix);
		cvjit::JitterMatrix results(out_matrix);

		// Make sure input is of proper format
		t_jit_err err = cvjit::Validate(x, input_image).type(_jit_sym_char).dimcount(2).min_dimsize(2);
		if (JIT_ERR_NONE == err)
		{
			try
			{
				// Convert Jitter matrix to OpenCV matrix
				cv::Mat src = input_image;

				if (src.channels() == 4) //If we have 4 channels ( Max Default ) we remove the Alpha layer
					cv::cvtColor(src, src, cv::COLOR_RGBA2BGR);

				cv::Mat dst = cv::dnn::blobFromImage(src, 1 / 255.F, x->resize_frame, cv::Scalar(), true, false, CV_32F);
				x->net.setInput(dst);

				// Compute
				std::vector<cv::Mat> outs;
				x->net.forward(outs, x->out_names);

				if (outs.size() == 0) {
					results.set_size(1);
					results.clear();
					return JIT_ERR_NONE;
				}

				results.set_size(outs.size());

				//Process outputs:
				std::vector<cv::Rect> boxes;
				std::vector<int> classIds;
				std::vector<float> confidences;

				for (size_t i = 0; i < outs.size(); ++i)
				{
					// Network produces output blob with a shape NxC where N is a number of detected objects and C is a number of classes + 4 where the first 4
					// numbers are [center_x, center_y, width, height]
					float* data = (float*)outs[i].data;
					for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
					{
						cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);

						cv::Point classIdPoint;
						double confidence;
						cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

						if (confidence > x->threshold)
						{
							int centerX = (int)(data[0] * src.cols);
							int centerY = (int)(data[1] * src.rows);
							int width = (int)(data[2] * src.cols);
							int height = (int)(data[3] * src.rows);
							int left = centerX - width / 2;
							int top = centerY - height / 2;

							classIds.push_back(classIdPoint.x);
							confidences.push_back((float)confidence);
							boxes.push_back(cv::Rect(left, top, width, height));
						}
					}
				}

				//check which objects passed the treshold value
				if (boxes.size() == 0) {
					results.set_size(1);
					results.clear();
					return JIT_ERR_NONE;
				}

				object_post((t_object*)x, "boxes(%d)", boxes.size());

				// Change the size of the output matrix
				results.set_size((long)boxes.size());

				// Get output Matrix data
				float* out_data = results.get_data<float>();

				zxing::Ref<zxing::Reader> reader(new zxing::MultiFormatReader);
				zxing::DecodeHints hints(zxing::DecodeHints::DEFAULT_HINT);
				hints.setTryHarder(true);

				for (cv::Rect& rect : boxes) {
					out_data[0] = (float)rect.x;
					out_data[1] = (float)rect.y;
					out_data[2] = (float)rect.width;
					out_data[3] = (float)rect.height;
					out_data += 4;

					try
					{
						cv::Mat barcode(src, rect); 
						time_t t = time(0);   // get time now
						cv::cvtColor(barcode, barcode, cv::COLOR_RGB2GRAY); //convert to grayscale

						std::string buffer = "test - " + std::to_string(long(std::time(nullptr))) + ".png";
						cv::imwrite(buffer.c_str(), barcode);

						zxing::Ref<zxing::LuminanceSource> source = MatSource::create(barcode);
						zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));
						zxing::Ref<zxing::BinaryBitmap> image(new zxing::BinaryBitmap(binarizer));

						zxing::Ref<zxing::Result> result = reader->decode(image, hints);

						object_post((t_object*)x, "Format:%s value:%s", zxing::BarcodeFormat::barcodeFormatNames[result->getBarcodeFormat()], result->getText()->getText().c_str());

					}
					catch (zxing::Exception& exception)
					{
						object_error((t_object*)x, "ZXing error: %s", exception.what());
					}
					object_post((t_object*)x, "Rect(%d,%d,%d,%d)", rect.x, rect.y, rect.width, rect.height);
				}

				return JIT_ERR_NONE;
			}
			catch (cv::Exception& exception)
			{
				object_error((t_object*)x, "OpenCV error: %s", exception.what());
			}
		}
		else
		{
			return err;
		}
	}
	return JIT_ERR_NONE;
}
