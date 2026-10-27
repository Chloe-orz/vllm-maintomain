"""云侧前缀缓存预留:协商边界钉住 + fail-closed 快照 + 布局校验。

LWD prefill-only 的 prefix caching 语义(对照集中式):

- 云侧 prompt 是占位零值(embeddings 由边侧上行),命中边界 H 在
  **协商时刻**由边侧预告的哈希链一次定死;边侧按 H 裁掉输入后,云侧
  不允许任何后续查询改变该边界——``get_computed_blocks`` 因此
  fail-closed:有协商记录的请求必须命中预留快照,查不到即报错,
  绝不退回"重新查询"得到不同边界。
- 预留是**短窗口**:reserve 钉住(touch)命中块防准入前被逐出,首次
  成功分配后由调度器 release,运行期保护交还原生请求 KV 链引用;
  抢占后命中块随原生 free 交还池子、可被逐出(复刻原生逐出语义)。
- 抢占重准入(设计 §5.3)复用同一预留表:调度器只读重查 H' 后经
  ``reserve_hit`` 钉住残块并登记快照;H' < 暂存地板 S 时先发
  refill 补发请求、挂闸门,补发完成后按钉住的 H' 快照准入。

本模块不 import vllm 运行时依赖(仅 TYPE_CHECKING/函数内延迟导入),
以便 CPU 单测直接加载。
"""

from __future__ import annotations

from math import lcm
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class _LwdPrefixReservation(NamedTuple):
    """单请求预留:fail-closed 快照 + touch 钉住的实体块。"""

    groups: tuple  # tuple(tuple(KVCacheBlock)),保持原生组布局供分配挂接
    hit_tokens: int
    pinned: list  # touch 过的实体块(已滤 null 槽),release 时 free_blocks


class LwdPrefixReservations:
    """命中块预留表(云引擎/调度器主线程独占)。

    入口四件:``reserve``(建请求协商)/ ``reserve_hit``(抢占重准入的
    只读重查结果钉住)/ ``get_computed_blocks``(KV 钩子的 fail-closed
    快照)/ ``release``(首分配后或终结兜底释放)。
    """

    def __init__(self, manager: KVCacheManager) -> None:
        self.manager = manager
        self._reservations: dict[str, _LwdPrefixReservation] = {}
        # 挂到 manager 上:kv_cache_manager.get_computed_blocks 的 LWD
        # 分支经此取快照;开关与对象同生同灭。
        manager.lwd_prefix_reservations = self
        manager.lwd_prefix_caching = True

    def reserve(self, request: Request) -> int:
        """建请求协商:查询最长命中 -> 钉住 -> 记快照,返回命中 token 数。

        命中长度上限 = num_prompt_tokens - 1(尾部至少保留 1 个 token
        参与计算,否则云侧无 logits 可采样首 token,沿用原生
        max_cache_hit_length 裁法)。skip_reading_prefix_cache 请求恒
        零命中、不钉块,但仍记快照(零命中承诺同样 fail-closed)。
        同时把命中边界写进 ``request.lwd_resume_cap_tokens`` —— 它是
        暂存地板 S(边侧实际发送过的行起点,命中请求 S=H),抢占重
        准入的统一路径以它为补发判据。
        """
        if request.request_id in self._reservations:
            raise ValueError(
                f"duplicate LWD prefix reservation: {request.request_id}"
            )
        if request.skip_reading_prefix_cache:
            groups: tuple = ()
            hit = 0
        else:
            groups, hit = self.manager.coordinator.find_longest_cache_hit(
                request.block_hashes,
                max_cache_hit_length=max(0, request.num_prompt_tokens - 1),
            )
        return self._pin_and_record(request, groups, hit, set_cap=True)

    def reserve_hit(
        self,
        request: Request,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
    ) -> int:
        """抢占重准入:调用方已只读重查出 H'=(groups, hit),此处只钉住
        残块并登记 fail-closed 快照(防准入/refill 期间再被逐出)。

        不改 ``lwd_resume_cap_tokens``:S 是暂存地板,由协商(初次
        reserve)与 refill 完成两个事件推进,重查命中不改变它。

        调用方须先做到达钳制(见调度器 ``_lwd_prep_prefix_readmission``):
        传入的 hit 不得超过该请求已到达行数,且 groups 须为 hit 覆盖的
        前缀块(可用 ``truncate_groups`` 截断)——钉超出钳制边界的块
        会让 stager 读指针前跳越过未写入行,破坏
        ``consumed <= recv_rows`` 不变量。
        """
        if request.request_id in self._reservations:
            raise ValueError(
                f"duplicate LWD prefix reservation: {request.request_id}"
            )
        return self._pin_and_record(request, groups, hit, set_cap=False)

    def truncate_groups(
        self, groups: tuple[Sequence[KVCacheBlock], ...], hit: int
    ) -> tuple:
        """把 find_longest_cache_hit 的组布局截到 hit 覆盖的前缀块。

        每组块大小都是 hash_block_size 的整数倍(布局校验保证),hit 已
        hash 块对齐,故每组恰截整块;null 槽原位保留(截断保持索引对齐)。
        hit=0 时全部截空(退化为无命中,不钉任何块)。
        """
        managers = self.manager.coordinator.single_type_managers
        return tuple(
            list(group[: hit // manager.block_size])
            for group, manager in zip(groups, managers)
        )

    def _pin_and_record(
        self, request: Request, groups: tuple, hit: int, set_cap: bool
    ) -> int:
        # 滑窗/状态缓存组会给不保留的位置垫 null 槽:只钉实体块,
        # 快照保留原生组布局(null 槽原位)供 allocate_slots 挂接。
        pinned = [
            block
            for group in groups
            for block in group
            if not getattr(block, "is_null", False)
        ]
        if pinned:
            self.manager.block_pool.touch(pinned)
        self._reservations[request.request_id] = _LwdPrefixReservation(
            groups=tuple(tuple(group) for group in groups),
            hit_tokens=hit,
            pinned=pinned,
        )
        if set_cap:
            request.lwd_resume_cap_tokens = hit
        return hit

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[tuple[Sequence[KVCacheBlock], ...], int]:
        """KV 钩子入口:有活跃预留返回协商/重查快照,否则 fail-closed。

        已协商(lwd_resume_cap_tokens 存在)但快照缺失 = 预留被提前
        释放或重准入预备漏跑——边侧已按旧边界裁掉输入,重新查询可能
        得到不同边界,必须报错而非静默放行。完全无协商记录(防御性
        兜底,正常 PC 路径不可达)才退回原生查询。
        """
        entry = self._reservations.get(request.request_id)
        if entry is not None:
            return entry.groups, entry.hit_tokens
        if getattr(request, "lwd_resume_cap_tokens", None) is not None:
            raise RuntimeError(
                f"LWD reserved prefix for {request.request_id} is "
                "unavailable; cannot consume trimmed embeddings "
                "(预留快照缺失,fail-closed)"
            )
        return self.manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )

    def has(self, request_id: str) -> bool:
        """是否有活跃预留(重准入预备的幂等判据)。"""
        return request_id in self._reservations

    def hit_of(self, request_id: str) -> int:
        """活跃预留的命中 token 数;无预留返 0(等价零命中)。"""
        entry = self._reservations.get(request_id)
        return entry.hit_tokens if entry is not None else 0

    def release(self, request_id: str) -> None:
        """释放预留(幂等):快照摘除,touch 钉住的块归还池子。

        调用点:首次成功分配后(短窗口语义,运行期保护交还原生 KV 链
        引用)与 ``_free_request`` 兜底(reserve->首分配窗口内的 abort/
        协商失败回滚/refill 在途 abort)。
        """
        entry = self._reservations.pop(request_id, None)
        if entry is not None and entry.pinned:
            self.manager.block_pool.free_blocks(entry.pinned)


def prefix_start(request: Request, reservations=None) -> int:
    """请求的调度起点:首分配前(num_computed==0)以预留命中为界,
    首分配后退回原生水位。

    窗口/可见性计算必须用它而非裸 num_computed:命中请求的
    ``recv_rows`` 从 H 起涨,起点取 0 会把永不到达的 [0, H) 行算进
    窗口。有活跃预留时以快照为准(抢占重准入 H' 可能 > 暂存地板);
    无预留退回协商边界,再退回 0。
    """
    if request.num_computed_tokens > 0:
        return request.num_computed_tokens
    if reservations is not None and reservations.has(request.request_id):
        return reservations.hit_of(request.request_id)
    return getattr(request, "lwd_resume_cap_tokens", 0)


class LwdPrefixCacheState(NamedTuple):
    """缓存模块返回的初始化结果,不持有或修改引擎对象。"""

    enabled: bool
    hash_block_size: int
    alignment_tokens: int = 0
    alignment_required: bool = False
    use_eagle: bool = False
    reservations: LwdPrefixReservations | None = None


def init_lwd_prefix_cache(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    manager: KVCacheManager,
    *,
    use_eagle: bool = False,
) -> LwdPrefixCacheState:
    """校验布局并创建预留管理对象,由调用方绑定调度器和 HELLO 状态。

    仅支持 FullAttention/MLA、SlidingWindow、Mamba align/all 组;PCP
    不支持(all-gather 布局与前缀命中边界不兼容)。不支持的布局 + PC
    开启 -> fail-fast(ValueError,引擎启动即拒绝)。

    Args:
        vllm_config: 云侧生效配置(读 enable_prefix_caching/PCP 宽度)。
        kv_cache_config: 调度器已定的 KV 组布局。
        manager: 云侧 KVCacheManager(预留表挂在它上面)。
        use_eagle: 云侧是否 MTP(eagle)模式(通告边侧,尾部规则以云为准)。

    Returns:
        LwdPrefixCacheState;enabled=False 时仅携带 hash_block_size 供
        HELLO 通告,不校验布局、不建预留表。

    Raises:
        ValueError: PC 开启但 manager 未开缓存 / 布局不支持 / PCP。
    """
    # 仅初始化时加载布局依赖,预留管理本身不依赖模型或引擎模块。
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    enabled = bool(vllm_config.cache_config.enable_prefix_caching)
    hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)[1]
    if not enabled:
        # 关闭缓存仍需通告 hash 粒度,但不校验缓存布局或创建预留对象。
        return LwdPrefixCacheState(enabled=False, hash_block_size=hash_block_size)
    if not manager.enable_caching:
        raise ValueError(
            "LWD prefix caching requires --enable-prefix-caching on cloud"
        )
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        raise ValueError(
            "LWD prefix caching does not support PCP "
            "(prefill_context_parallel_size > 1)"
        )
    specs = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs.extend(
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
    if not specs or any(
        not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
        or (
            isinstance(spec, MambaSpec)
            and spec.mamba_cache_mode not in ("align", "all")
        )
        for spec in specs
    ):
        raise ValueError(
            "LWD prefix caching currently supports "
            "Full Attention/MLA, sliding-window and Mamba align/all groups only"
        )
    coordinator = manager.coordinator
    alignment_tokens = lcm(
        coordinator.scheduler_block_size,
        getattr(coordinator, "lcm_block_size", coordinator.scheduler_block_size),
    )
    alignment_required = any(
        isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
        for spec in specs
    )
    return LwdPrefixCacheState(
        enabled=True,
        hash_block_size=hash_block_size,
        alignment_tokens=alignment_tokens,
        alignment_required=alignment_required,
        use_eagle=use_eagle,
        reservations=LwdPrefixReservations(manager),
    )
