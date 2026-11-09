"""Small Tk window for live Jev request status."""

import time
import tkinter as tk
from tkinter import ttk


class JevStatusWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Jev 调用状态")
        self.root.geometry("700x305")
        self.root.minsize(560, 275)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar(value="等待 Jev 请求")
        self.question = tk.StringVar(value="当前 Jev 问题：等待搜索开始")
        self.state = tk.StringVar(value="当前状态：—")
        self.returned = tk.StringVar(value="最近一次 Jev 返回值：尚未返回")
        self.heuristic = tk.StringVar(value="搜索使用的 h：—")
        self.derived = tk.StringVar(value="换算值：—")
        self.elapsed = tk.StringVar(value="本次调用耗时：—")
        self.total = tk.StringVar(value="API 累计请求：0")
        self.estimate = tk.StringVar(value="预计剩余：—")
        self.closed = False
        self._started = None
        self._run_started = None
        self._expected_requests = None
        self._request_number = 0

        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.status, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        for variable in (
            self.question,
            self.state,
            self.returned,
            self.heuristic,
            self.derived,
            self.elapsed,
            self.total,
            self.estimate,
        ):
            ttk.Label(frame, textvariable=variable, wraplength=580).pack(
                anchor="w", pady=2
            )

    @staticmethod
    def _brief_question(question):
        text = " ".join(str(question).split())
        lowered = text.lower()
        if "remaining coins" in lowered or "remaining coin" in lowered:
            return "估算收集完所有剩余硬币的最少步数"
        if "overall judgement" in lowered or "holistic state-quality" in lowered:
            return "综合评估局面：收集进度、生存风险、装备与行动选择"
        if "how promising" in lowered or "promising" in lowered:
            return "判断该位置是否适合走在最短路线中"
        if "manhattan lower bound" in lowered:
            return "估算实际剩余路程是曼哈顿距离的几倍"
        if "shortest path" in lowered or "shortest route" in lowered:
            return "估算到目标的最短剩余步数"
        if len(text) > 100:
            return text[:97] + "..."
        return text

    @staticmethod
    def _brief_states(candidates):
        summaries = []
        for candidate in candidates[:3]:
            state = candidate.get("state")
            if (
                isinstance(state, tuple)
                and len(state) == 2
                and hasattr(state[1], "count")
            ):
                try:
                    summaries.append(
                        "位置 %s，剩余硬币 %d 枚" % (state[0], state[1].count())
                    )
                    continue
                except Exception:
                    pass
            summaries.append("位置 %s" % (state,))
        if len(candidates) > len(summaries):
            summaries.append("另有 %d 个状态" % (len(candidates) - len(summaries)))
        return "；".join(summaries) or "—"

    @staticmethod
    def _format_values(values):
        values = list(values)
        if not values:
            return "无成功返回值"
        formatted = ["%.3f" % float(value) for value in values[:6]]
        if len(values) == 1:
            return formatted[0]
        suffix = " …" if len(values) > 6 else ""
        return "%d 个值：%s%s" % (len(values), ", ".join(formatted), suffix)

    def _refresh(self):
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def begin_request(
        self,
        question,
        candidates,
        api_requests=0,
        state_summary=None,
        expected_requests=None,
    ):
        if self.closed:
            return
        self._request_number += 1
        self._started = time.perf_counter()
        if self._run_started is None:
            self._run_started = self._started
        if expected_requests is not None:
            self._expected_requests = max(0, int(expected_requests))
        self.status.set("正在等待 Jev 返回（请求 #%d）" % self._request_number)
        self.question.set("当前 Jev 问题：%s" % self._brief_question(question))
        summary = state_summary or self._brief_states(candidates)
        self.state.set("当前状态：%s" % summary)
        # Keep the previous result visible while the next search-state request
        # is in flight. Q9 can issue calls back-to-back, so clearing this here
        # made each completed value disappear almost immediately.
        self.heuristic.set("搜索使用的 h：—")
        self.elapsed.set("本次调用耗时：0.00 秒")
        self.update_progress(api_requests)
        self._refresh()

    def pump(self, api_requests=None):
        if self.closed:
            return
        if api_requests is not None:
            self.update_progress(api_requests)
        if self._started is not None:
            duration = time.perf_counter() - self._started
            self.elapsed.set("本次调用耗时：%.2f 秒（进行中）" % duration)
        self._refresh()

    def set_values(self, label, values, duration, api_requests=None):
        if self.closed:
            return
        self._started = None
        if api_requests is not None:
            self.update_progress(api_requests)
        self.status.set("Jev 已返回")
        self.returned.set(
            "最近一次 Jev 返回值（%s）：%s"
            % (label, self._format_values(values))
        )
        self.elapsed.set("本次调用耗时：%.2f 秒" % duration)
        self._refresh()

    def set_heuristic(self, value):
        if self.closed:
            return
        self.heuristic.set("搜索使用的 h：%.3f" % float(value))
        self._refresh()

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds + 0.5))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return "%d小时 %02d分" % (hours, minutes)
        if minutes:
            return "%d分 %02d秒" % (minutes, seconds)
        return "%d秒" % seconds

    def update_progress(self, api_requests):
        if self.closed:
            return
        api_requests = max(0, int(api_requests))
        self.total.set("API 累计请求：%d" % api_requests)
        if self._expected_requests is None:
            self.estimate.set("预计剩余：未设置调用总量")
            return
        if self._run_started is None or api_requests == 0:
            self.estimate.set(
                "预计剩余：等待调用数据（目标约 %d 次）"
                % self._expected_requests
            )
            return

        elapsed = max(0.001, time.perf_counter() - self._run_started)
        calls_per_second = api_requests / elapsed
        remaining = max(0, self._expected_requests - api_requests)
        if remaining == 0:
            eta = "已达到估算调用量"
        else:
            eta = "约剩 %s" % self._format_duration(
                remaining / calls_per_second
            )
        self.estimate.set(
            "预计剩余：%s；均速 %.2f 次/秒；进度 %d/%d（目标约）"
            % (
                eta,
                calls_per_second,
                api_requests,
                self._expected_requests,
            )
        )

    def set_derived(self, label, value):
        if self.closed:
            return
        self.derived.set("%s：%.3f" % (label, float(value)))
        self._refresh()

    def set_failure(self, message, duration, api_requests=None):
        if self.closed:
            return
        self._started = None
        if api_requests is not None:
            self.update_progress(api_requests)
        self.status.set("Jev 请求失败，搜索将使用回退值")
        if self.returned.get() == "最近一次 Jev 返回值：尚未返回":
            self.returned.set("最近一次 Jev 返回值：无（%s）" % str(message)[:100])
        self.elapsed.set("本次调用耗时：%.2f 秒" % duration)
        self._refresh()

    def finish_search(self, stats):
        if self.closed:
            return
        self._started = None
        self.status.set("搜索计算完成；关闭此窗口后程序继续")
        if "requests" in stats and "api_requests" not in stats:
            summary = (
                "API 请求：%d；成功：%d；评估状态：%d；缓存命中：%d；"
                "回退：%d；API 累计耗时：%.2f 秒"
                % (
                    stats.get("requests", 0),
                    stats.get("successful_requests", 0),
                    stats.get("evaluations", 0),
                    stats.get("cache_hits", 0),
                    stats.get("fallback_states", 0),
                    stats.get("api_seconds", 0.0),
                )
            )
        else:
            summary = (
                "API 累计请求：%d；已评分状态：%d；缓存命中：%d；"
                "API 累计耗时：%.2f 秒"
                % (
                    stats.get("api_requests", 0),
                    stats.get("scored_states", 0),
                    stats.get("cache_hits", 0),
                    stats.get("api_seconds", 0.0),
                )
            )
        self.total.set(summary)
        self._refresh()
        if not self.closed:
            self.root.mainloop()
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass
