"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _Reason:
    ValueType = typing.NewType('ValueType', builtins.int)
    V: typing_extensions.TypeAlias = ValueType
class _ReasonEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_Reason.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    UNKNOWN: _Reason.ValueType  # 0
    RECONNECT: _Reason.ValueType  # 1
    POWER_DISCONNECTED: _Reason.ValueType  # 2
    WIFI_UNAVAILABLE: _Reason.ValueType  # 3
    ACK: _Reason.ValueType  # 4
class Reason(_Reason, metaclass=_ReasonEnumTypeWrapper):
    pass

UNKNOWN: Reason.ValueType  # 0
RECONNECT: Reason.ValueType  # 1
POWER_DISCONNECTED: Reason.ValueType  # 2
WIFI_UNAVAILABLE: Reason.ValueType  # 3
ACK: Reason.ValueType  # 4
global___Reason = Reason


class Parameters(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TENSORS_FIELD_NUMBER: builtins.int
    TENSOR_TYPE_FIELD_NUMBER: builtins.int
    tensors: builtins.bytes
    tensor_type: typing.Text
    def __init__(self,
        *,
        tensors: builtins.bytes = ...,
        tensor_type: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tensor_type",b"tensor_type","tensors",b"tensors"]) -> None: ...
global___Parameters = Parameters

class ServerMessage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Reconnect(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        SECONDS_FIELD_NUMBER: builtins.int
        seconds: builtins.int
        def __init__(self,
            *,
            seconds: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["seconds",b"seconds"]) -> None: ...

    class GetParameters(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        def __init__(self,
            ) -> None: ...

    class FitIns(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        class ConfigEntry(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            KEY_FIELD_NUMBER: builtins.int
            VALUE_FIELD_NUMBER: builtins.int
            key: typing.Text
            @property
            def value(self) -> global___Scalar: ...
            def __init__(self,
                *,
                key: typing.Text = ...,
                value: typing.Optional[global___Scalar] = ...,
                ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

        PARAMETERS_FIELD_NUMBER: builtins.int
        CONFIG_FIELD_NUMBER: builtins.int
        @property
        def parameters(self) -> global___Parameters: ...
        @property
        def config(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Scalar]: ...
        def __init__(self,
            *,
            parameters: typing.Optional[global___Parameters] = ...,
            config: typing.Optional[typing.Mapping[typing.Text, global___Scalar]] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["parameters",b"parameters"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["config",b"config","parameters",b"parameters"]) -> None: ...

    class EvaluateIns(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        class ConfigEntry(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            KEY_FIELD_NUMBER: builtins.int
            VALUE_FIELD_NUMBER: builtins.int
            key: typing.Text
            @property
            def value(self) -> global___Scalar: ...
            def __init__(self,
                *,
                key: typing.Text = ...,
                value: typing.Optional[global___Scalar] = ...,
                ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

        PARAMETERS_FIELD_NUMBER: builtins.int
        CONFIG_FIELD_NUMBER: builtins.int
        @property
        def parameters(self) -> global___Parameters: ...
        @property
        def config(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Scalar]: ...
        def __init__(self,
            *,
            parameters: typing.Optional[global___Parameters] = ...,
            config: typing.Optional[typing.Mapping[typing.Text, global___Scalar]] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["parameters",b"parameters"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["config",b"config","parameters",b"parameters"]) -> None: ...

    RECONNECT_FIELD_NUMBER: builtins.int
    GET_PARAMETERS_FIELD_NUMBER: builtins.int
    FIT_INS_FIELD_NUMBER: builtins.int
    EVALUATE_INS_FIELD_NUMBER: builtins.int
    @property
    def reconnect(self) -> global___ServerMessage.Reconnect: ...
    @property
    def get_parameters(self) -> global___ServerMessage.GetParameters: ...
    @property
    def fit_ins(self) -> global___ServerMessage.FitIns: ...
    @property
    def evaluate_ins(self) -> global___ServerMessage.EvaluateIns: ...
    def __init__(self,
        *,
        reconnect: typing.Optional[global___ServerMessage.Reconnect] = ...,
        get_parameters: typing.Optional[global___ServerMessage.GetParameters] = ...,
        fit_ins: typing.Optional[global___ServerMessage.FitIns] = ...,
        evaluate_ins: typing.Optional[global___ServerMessage.EvaluateIns] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["evaluate_ins",b"evaluate_ins","fit_ins",b"fit_ins","get_parameters",b"get_parameters","msg",b"msg","reconnect",b"reconnect"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluate_ins",b"evaluate_ins","fit_ins",b"fit_ins","get_parameters",b"get_parameters","msg",b"msg","reconnect",b"reconnect"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["msg",b"msg"]) -> typing.Optional[typing_extensions.Literal["reconnect","get_parameters","fit_ins","evaluate_ins"]]: ...
global___ServerMessage = ServerMessage

class ClientMessage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Disconnect(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        REASON_FIELD_NUMBER: builtins.int
        reason: global___Reason.ValueType
        def __init__(self,
            *,
            reason: global___Reason.ValueType = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["reason",b"reason"]) -> None: ...

    class ParametersRes(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        PARAMETERS_FIELD_NUMBER: builtins.int
        @property
        def parameters(self) -> global___Parameters: ...
        def __init__(self,
            *,
            parameters: typing.Optional[global___Parameters] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["parameters",b"parameters"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["parameters",b"parameters"]) -> None: ...

    class FitRes(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        class MetricsEntry(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            KEY_FIELD_NUMBER: builtins.int
            VALUE_FIELD_NUMBER: builtins.int
            key: typing.Text
            @property
            def value(self) -> global___Scalar: ...
            def __init__(self,
                *,
                key: typing.Text = ...,
                value: typing.Optional[global___Scalar] = ...,
                ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

        PARAMETERS_FIELD_NUMBER: builtins.int
        NUM_EXAMPLES_FIELD_NUMBER: builtins.int
        NUM_EXAMPLES_CEIL_FIELD_NUMBER: builtins.int
        FIT_DURATION_FIELD_NUMBER: builtins.int
        METRICS_FIELD_NUMBER: builtins.int
        @property
        def parameters(self) -> global___Parameters: ...
        num_examples: builtins.int
        num_examples_ceil: builtins.int
        fit_duration: builtins.float
        @property
        def metrics(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Scalar]: ...
        def __init__(self,
            *,
            parameters: typing.Optional[global___Parameters] = ...,
            num_examples: builtins.int = ...,
            num_examples_ceil: builtins.int = ...,
            fit_duration: builtins.float = ...,
            metrics: typing.Optional[typing.Mapping[typing.Text, global___Scalar]] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["parameters",b"parameters"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["fit_duration",b"fit_duration","metrics",b"metrics","num_examples",b"num_examples","num_examples_ceil",b"num_examples_ceil","parameters",b"parameters"]) -> None: ...

    class EvaluateRes(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        class MetricsEntry(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            KEY_FIELD_NUMBER: builtins.int
            VALUE_FIELD_NUMBER: builtins.int
            key: typing.Text
            @property
            def value(self) -> global___Scalar: ...
            def __init__(self,
                *,
                key: typing.Text = ...,
                value: typing.Optional[global___Scalar] = ...,
                ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

        NUM_EXAMPLES_FIELD_NUMBER: builtins.int
        LOSS_FIELD_NUMBER: builtins.int
        ACCURACY_FIELD_NUMBER: builtins.int
        METRICS_FIELD_NUMBER: builtins.int
        num_examples: builtins.int
        loss: builtins.float
        accuracy: builtins.float
        @property
        def metrics(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Scalar]: ...
        def __init__(self,
            *,
            num_examples: builtins.int = ...,
            loss: builtins.float = ...,
            accuracy: builtins.float = ...,
            metrics: typing.Optional[typing.Mapping[typing.Text, global___Scalar]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["accuracy",b"accuracy","loss",b"loss","metrics",b"metrics","num_examples",b"num_examples"]) -> None: ...

    DISCONNECT_FIELD_NUMBER: builtins.int
    PARAMETERS_RES_FIELD_NUMBER: builtins.int
    FIT_RES_FIELD_NUMBER: builtins.int
    EVALUATE_RES_FIELD_NUMBER: builtins.int
    @property
    def disconnect(self) -> global___ClientMessage.Disconnect: ...
    @property
    def parameters_res(self) -> global___ClientMessage.ParametersRes: ...
    @property
    def fit_res(self) -> global___ClientMessage.FitRes: ...
    @property
    def evaluate_res(self) -> global___ClientMessage.EvaluateRes: ...
    def __init__(self,
        *,
        disconnect: typing.Optional[global___ClientMessage.Disconnect] = ...,
        parameters_res: typing.Optional[global___ClientMessage.ParametersRes] = ...,
        fit_res: typing.Optional[global___ClientMessage.FitRes] = ...,
        evaluate_res: typing.Optional[global___ClientMessage.EvaluateRes] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["disconnect",b"disconnect","evaluate_res",b"evaluate_res","fit_res",b"fit_res","msg",b"msg","parameters_res",b"parameters_res"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["disconnect",b"disconnect","evaluate_res",b"evaluate_res","fit_res",b"fit_res","msg",b"msg","parameters_res",b"parameters_res"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["msg",b"msg"]) -> typing.Optional[typing_extensions.Literal["disconnect","parameters_res","fit_res","evaluate_res"]]: ...
global___ClientMessage = ClientMessage

class Scalar(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    DOUBLE_FIELD_NUMBER: builtins.int
    SINT64_FIELD_NUMBER: builtins.int
    BOOL_FIELD_NUMBER: builtins.int
    STRING_FIELD_NUMBER: builtins.int
    BYTES_FIELD_NUMBER: builtins.int
    double: builtins.float
    sint64: builtins.int
    """float float = 2;
    int32 int32 = 3;
    int64 int64 = 4;
    uint32 uint32 = 5;
    uint64 uint64 = 6;
    sint32 sint32 = 7;
    """

    bool: builtins.bool
    """fixed32 fixed32 = 9;
    fixed64 fixed64 = 10;
    sfixed32 sfixed32 = 11;
    sfixed64 sfixed64 = 12;
    """

    string: typing.Text
    bytes: builtins.bytes
    def __init__(self,
        *,
        double: builtins.float = ...,
        sint64: builtins.int = ...,
        bool: builtins.bool = ...,
        string: typing.Text = ...,
        bytes: builtins.bytes = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bool",b"bool","bytes",b"bytes","double",b"double","scalar",b"scalar","sint64",b"sint64","string",b"string"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bool",b"bool","bytes",b"bytes","double",b"double","scalar",b"scalar","sint64",b"sint64","string",b"string"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["scalar",b"scalar"]) -> typing.Optional[typing_extensions.Literal["double","sint64","bool","string","bytes"]]: ...
global___Scalar = Scalar
