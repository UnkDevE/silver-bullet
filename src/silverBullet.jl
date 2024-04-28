module silverBullet

using Pkg
Pkg.activate(".")

if length(ARGS) != 3
	println("Needs three arguments:
			 the models struct name in JLD2 save
			 the file of saved state (Flux) of the model struct in JDL2 
			 and the test data in a folder ready to be containerized")
	exit()
end

model_struct, model_file, test_data = ARGS

using Flux, JLD2
model = JLD2.load(model_file, model_struct)

using MLDatasets, DataFrames, FileIO

data = FileDataset(test_data)

# batch test into features
features = data.metadata["feature_names"]

feature_buckets = map(str -> filter(:feature = str, data), features)

# predict model from buckets
predict_features = map(fture -> model(fture), feature_buckets)

# cacluate length of inverse
@show grads = Flux.gradient(model)

end # module silverBullet
