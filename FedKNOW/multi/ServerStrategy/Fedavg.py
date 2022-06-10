import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import pickle

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
class OurFed(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn = None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            accept_failures: bool = True,
            initial_parameters = None,
    ) -> None:
        super(OurFed, self).__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters
        )
        self.kb = []
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager
    ) :
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        config['round'] = rnd

        print("round is now",rnd)
        kb_converted = []
        kb_converted_string = ""
        if(len(self.kb) > 0):
            for tensor in range(len(self.kb[0])):
                internal = []
                for client in range(len(self.kb)):
                    internal.append(self.kb[client][tensor])
                kb_converted.append(np.stack(internal,axis=-1))
            kb_converted_string = pickle.dumps(kb_converted)
            self.kb = []
        config['kb'] = kb_converted_string

        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        config['round'] = rnd
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}



        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = weights_to_parameters(aggregate(weights_results))
        # print("--the length of data is---- ")
        kb = []
        for client,fitres in results:
            print(client.cid)
            print('kb' in fitres.metrics)
        for _,fitres in results:
            if (fitres.metrics['kb'] != ""):
                kb.append(pickle.loads(fitres.metrics['kb']))

        self.kb = kb
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif rnd == 1:  # Only log this warning once
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
