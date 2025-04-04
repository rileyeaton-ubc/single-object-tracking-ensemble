function params = init_default_params(params)

% Initialize default parameters
default_params.use_gpu = false;
default_params.gpu_id = [];
default_params.confidence_threshold = 0.10;
default_params.redetection_threshold = 0.13;
default_params.is_currently_lost = false;

def_param_names = fieldnames(default_params);
for k = 1:numel(def_param_names)
    param_name = def_param_names{k};
    if ~isfield(params, param_name)
        params.(param_name) = default_params.(param_name);
    end
end

params.visualization = params.visualization;