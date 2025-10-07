import redis
from dataclasses import dataclass, field


@dataclass
class RedisServerParams:
    port: str = "6379"
    password: str = "ippc123456"
    host_ip: str = "10.162.12.223"
    name: str = "server_1"


@dataclass
class ChannelMappingParams:
    server_name: str = "server_1"
    channel_name: str = "channel"


@dataclass
class RedisConfig:
    servers: RedisServerParams = field(default_factory=RedisServerParams)
    ch_edge_control: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_plant_trajectory_segment: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_weights: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_ready_update: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_plant_reset: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_mode: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_trajectory: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_training_steps: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_patch_update: ChannelMappingParams = field(default_factory=ChannelMappingParams)
    ch_edge_patch_gain: ChannelMappingParams = field(default_factory=ChannelMappingParams)


class RedisConnection:
    def __init__(self, params: RedisConfig):
        self.params = params
        self.pools \
            = {self.params.servers.name: redis.ConnectionPool(host=self.params.servers.host_ip,
                                                              port=self.params.servers.port,
                                                              password=self.params.servers.password)}

        self.conns = {name: redis.Redis(connection_pool=pool) for (name, pool) in self.pools.items()}

    def publish(self, channel: ChannelMappingParams, message):
        self.conns[channel.server_name].publish(channel.channel_name, message)

    def subscribe(self, channel: ChannelMappingParams):
        substates = self.conns[channel.server_name].pubsub()
        substates.subscribe(channel.channel_name)
        substates.parse_response()
        return substates
