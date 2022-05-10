import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import NUM_INT_STEPS

def RELU(x):
    return x.relu()
def POOL(x):
    return F.max_pool2d(x,2)
def POOL1d(x):
    return F.max_pool1d(x,2)



class ResidualNN(nn.Module):
    def __init__(self, D_in, hidden_size, hidden_layers=2):
        super(ResidualNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.D_in = D_in

        for i in range(self.hidden_layers + 1):
            if i == 0:
                D_in = self.D_in
            else:
                D_in = hidden_size
            if i == self.hidden_layers:
                D_out = self.D_in
            else:
                D_out = hidden_size

            fc = nn.Linear(D_in, D_out, bias=False)
            bn = nn.BatchNorm1d(D_out)
            pr_relu = nn.PReLU()

            # fc gets all 0s when initialized
            fc.weight.data.fill_(0.00)

            setattr(self, f'fc_{i+1}', fc)
            setattr(self, f'bn_{i+1}', bn)
            setattr(self, f'pr_relu_{i+1}', pr_relu)

    def forward(self, x):
        z = x

        for i in range(self.hidden_layers + 1):
            fc = getattr(self, f'fc_{i+1}')
            bn = getattr(self, f'bn_{i+1}')
            pr_relu = getattr(self, f'pr_relu_{i+1}')

            z = pr_relu(bn(fc(z)))

        # residual skip connection
        z = z + x

        return z

class DefaultNN(nn.Module):
    def __init__(self, D_in, hidden_size):
        super(DefaultNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(D_in, hidden_size, bias=False),
                                 nn.BatchNorm1d(hidden_size), nn.PReLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=False),
                                 nn.BatchNorm1d(hidden_size), nn.PReLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=False),
                                 nn.BatchNorm1d(hidden_size), nn.PReLU())

    def forward(self, x):
        return self.net(x)

class MukundMDNHelper(nn.Module):
    def __init__(self, D_in, hidden_size, init_type, n_components):
        super(MukundMDNHelper,self).__init__()
        if init_type in ['default', 'fixed']:
            self.fc_in = DefaultNN(D_in, hidden_size)
            pi_layer = nn.Linear(hidden_size, n_components)
            mu_layer = nn.Linear(hidden_size, n_components)
            sigma_layer = nn.Linear(hidden_size, n_components)
            # well-spaced mu layers between [-3, 3] (normalized data)
            mu_layer.weight.data.fill_(0.0)
            mu_layer.bias.data = torch.linspace(-3, 3, n_components)
            # if fixed init, then change pi and sigma layers
            if init_type == 'fixed':
                # uniform pi
                pi_layer.weight.data.fill_(0.0)
                pi_layer.bias.data = torch.ones(n_components) /  n_components

                # std deviation = 1
                self.y_sigma = 1
                sigma_layer.weight.data.fill_(0.0)
                sigma_layer.bias.data = torch.ones(
                    n_components) * torch.log(
                    torch.exp(torch.ones(1) / self.y_sigma) - 1)

        if init_type == 'residual':
            self.fc_in = ResidualNN(D_in, hidden_size)
            pi_layer = nn.Linear(D_in, n_components)
            mu_layer = nn.Linear(D_in, n_components)
            sigma_layer = nn.Linear(D_in, n_components)
            # set mu such that one of the centers is the same as the jth feature
            mu_layer.weight.data = torch.zeros(n_components, D_in)
            mu_layer.bias.data = torch.cat(
                [torch.zeros(1), torch.linspace(-3, 3, n_components - 1)])
            # uniform pi
            pi_layer.weight.data.fill_(0.0)
            pi_layer.bias.data = torch.ones(
                n_components) / n_components

        # define parameter transforms
        ## pi
        self.pi_transform = nn.Sequential(pi_layer)
        ## mu
        self.mu_transform = nn.Sequential(mu_layer)
        ## sigma
        self.sigma_transform = nn.Sequential(sigma_layer)
    def forward(self,src):
        z = self.fc_in(src)
        # get MDN parameters
        pi = self.pi_transform(z.clamp(-1e3, 1e3)).unsqueeze(1)
        mu = self.mu_transform(z).clamp(-30, 30).unsqueeze(1)
        pre_log_sigma = self.sigma_transform(z).clamp(1e-2, 1e2).unsqueeze(1)
        pred = torch.cat([pi, mu, pre_log_sigma], dim=1)
        return pred

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)


def safe_inverse_softplus(x):
    return torch.log(torch.exp(x) - 1 + 1e-6)

def inverse_softplus_grad(x):
    return torch.exp(x) / (torch.exp(x) - 1 + 1e-6)

def logsumexp(a, dim, b):
    # support subtraction in logsumexp
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-6)
    out += a_max
    return out



class SurvivalMDN_Dist():
    def __init__(self, pred_params):
        self.pred_params = pred_params

    def log_prob(self, times):
        inv_softplus_times = safe_inverse_softplus(times)  # batch size
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)  # (batch size, 1)
        ws, mus, sigmas = self.pred_params_to_parameters(self.pred_params)
        num_components = ws.shape[1]
        normal_dists = torch.distributions.Normal(mus, sigmas)
        repeated_inv_softplus_times = inv_softplus_times.repeat(1, num_components)  # (batch_size, num_components)
        log_normal_pdfs = normal_dists.log_prob(repeated_inv_softplus_times)  # (batch_size, num_components)
        log_pdfs = logsumexp(a=log_normal_pdfs, dim=-1, b=ws).squeeze()  # batch_size
        log_grad = torch.log(inverse_softplus_grad(times)) #batch_size
        return log_pdfs + log_grad

    def cdf(self, times):
        inv_softplus_times = safe_inverse_softplus(times)  # batch size
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)  # (batch size, 1)
        ws, mus, sigmas = self.pred_params_to_parameters(self.pred_params)
        num_components = ws.shape[1]
        if len(times.shape) == 1:
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, num_components)  # (batch_size, num_components)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)  # (batch_size, num_components)
            cdfs = ws * log_normal_cdfs  # (batch_size, num_components)
            cdfs = cdfs.sum(-1)  # batch_size
        elif len(times.shape) == 2:
            time_len = times.shape[-1]
            mus = mus.unsqueeze(1).repeat(1, time_len, 1) # (batch size, time_len, num components)
            sigmas = sigmas.unsqueeze(1).repeat(1, time_len, 1) # (batch size, time_len, num components)
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, num_components)  # (batch_size, time_len, num_components)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)  # (batch_size, time_len, num_components)
            ws = ws.unsqueeze(1) # (batch_size, 1, num_components)
            cdfs = ws * log_normal_cdfs  # (batch_size, time_len, num_components)
            cdfs = cdfs.sum(-1)  # batch_size, time_len
            # Fix cdfs > 1 due to precision lost in ws
            cdfs[cdfs > 1.] = 1.
        else:
            assert  False, "times shape not supported"
        return cdfs

    def pred_params_to_parameters(self, pred_params):
        ws = self.pred_params[:, 0, :]  # (batch size, num components)
        ws = F.softmax(ws, dim=-1)  # (batch size, num components)
        mus = self.pred_params[:, 1, :]  # (batch size, num components)
        sigmas = self.pred_params[:, 2, :]  # (batch size, num components)
        sigmas = F.softplus(sigmas)
        return ws, mus, sigmas

class MDNModel(nn.Module):
    def __init__(self, model_config=None, feature_size=None):
        super(MDNModel, self).__init__()
        hidden_size = model_config['hidden_size']
        init_type = model_config['init_type']
        num_components = model_config['num_components']
        self.model = MukundMDNHelper(feature_size, hidden_size, init_type, num_components)
        self.set_last_eval(False)

    def set_last_eval(self, last_eval=True):
        self.last_eval = last_eval

    def forward(self, inputs):
        device = next(self.parameters()).device
        t = inputs["t"]
        features = inputs["features"]
        pred_params = self.model(features)
        dist = SurvivalMDN_Dist(pred_params)
        outputs = {}
        survival_func = 1 - dist.cdf(t)
        pdf = torch.exp(dist.log_prob(t))
        outputs["Lambda"] = -torch.log(survival_func + 1e-6)
        outputs["lambda"] = pdf / survival_func
        if not self.training:
            if self.last_eval and "eval_t" in inputs:
                ones = torch.ones_like(inputs["t"])
                # Eval for time-dependent C-index
                outputs["t"] = inputs["t"]
                outputs["eval_t"] = inputs["eval_t"] # eval_len
                batch_size = features.shape[0]
                eval_t = inputs["eval_t"].unsqueeze(0).repeat(batch_size, 1) # batch_size, eval_len
                cdf = dist.cdf(eval_t) # batch_size, eval_len
                survival_func = 1 - cdf # batch_size, eval_len
                cum_hazard = - torch.log(survival_func) # batch_size, eval_len
                outputs["cum_hazard_seqs"] = cum_hazard.transpose(1, 0) # eval_len, batch_size

                # Eval for Brier Score
                t_min = inputs["t_min"]
                t_max = inputs["t_max"]
                t = torch.linspace(
                    t_min, t_max, NUM_INT_STEPS, dtype=t_min.dtype,
                    device=device)
                eval_t = t.unsqueeze(0).repeat(batch_size, 1) # batch_size, eval_len
                cdf = dist.cdf(eval_t) # batch_size, eval_len
                survival_func = 1 - cdf # batch_size, eval_len
                outputs["survival_seqs"] = survival_func.transpose(1, 0) # eval_len, batch_size

                for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    t_min = inputs["t_min"]
                    t_max = inputs["t_max_{}".format(eps)]
                    t = torch.linspace(
                        t_min, t_max, NUM_INT_STEPS, dtype=t_min.dtype,
                        device=device)
                    eval_t = t.unsqueeze(0).repeat(batch_size, 1)  # batch_size, eval_len
                    cdf = dist.cdf(eval_t)  # batch_size, eval_len
                    survival_func = 1 - cdf  # batch_size, eval_len
                    outputs["survival_seqs_{}".format(eps)] = survival_func.transpose(1, 0) # eval_len, batch_size

            if "t_q25" in inputs:
                outputs["t"] = inputs["t"]
                for q in ["q25", "q50", "q75"]:
                    t = inputs["t_%s" % q]
                    survival_func = 1 - dist.cdf(t)
                    outputs["Lambda_%s" % q] = -torch.log(survival_func + 1e-6)

        return outputs
