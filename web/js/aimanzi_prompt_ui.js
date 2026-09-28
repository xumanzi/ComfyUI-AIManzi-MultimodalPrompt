import { app } from "/scripts/app.js";

const NODE = "AIManziMultimodalPrompt";
const TEMPLATE_NODE = "AIManziReadText";
const MAX_IMAGE_INPUTS = 10;
const SEED_MAX = 0xFFFFFFFF;
const SEED_MODES = ["随机", "增加", "减少", "固定"];

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetVisible(node, name, visible) {
    const widget = getWidget(node, name);
    if (!widget) return;
    if (widget.__aimanziOriginalType === undefined) {
        widget.__aimanziOriginalType = widget.type;
        widget.__aimanziOriginalComputeSize = widget.computeSize;
    }
    widget.hidden = !visible;
    widget.type = visible ? widget.__aimanziOriginalType : "hidden";
    widget.computeSize = visible ? widget.__aimanziOriginalComputeSize : () => [0, -4];
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach((linked) => {
            linked.hidden = !visible;
            linked.type = visible ? (linked.__aimanziOriginalType ?? linked.type) : "hidden";
        });
    }
}

function refreshLayout(node) {
    const wanted = node.computeSize();
    node.setSize([Math.max(460, wanted[0]), Math.max(wanted[1] + 24, 300)]);
    app.graph.setDirtyCanvas(true, true);
}

function normalizeSeedMode(value) {
    const mode = String(value ?? "").toLowerCase();
    if (["随机", "random", "randomize"].includes(mode)) return "随机";
    if (["增加", "increment"].includes(mode)) return "增加";
    if (["减少", "decrement"].includes(mode)) return "减少";
    return "固定";
}

function randomSeed(previous) {
    const values = new Uint32Array(1);
    if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(values);
    else values[0] = Math.floor(Math.random() * (SEED_MAX + 1));
    let value = Number(values[0]);
    if (value === Number(previous)) value = (value + 1) >>> 0;
    return value;
}

function ensureSeedControl(node) {
    const seed = getWidget(node, "随机种子");
    if (!seed) return;
    let mode = getWidget(node, "种子控制");
    if (mode) {
        mode.value = normalizeSeedMode(mode.value);
        return;
    }

    // ComfyUI's automatically linked control_after_generate is unreliable on
    // this dynamically laid-out node: some frontend versions neither update the
    // visible number nor serialize the changed value. Replace only this node's
    // linked control with an explicit, persisted mode widget.
    const native = seed.linkedWidgets?.find((widget) => widget.name === "control_after_generate")
        ?? getWidget(node, "control_after_generate");
    const initialMode = normalizeSeedMode(native?.value ?? "fixed");
    if (native) {
        const nativeIndex = node.widgets.indexOf(native);
        if (nativeIndex >= 0) node.widgets.splice(nativeIndex, 1);
    }
    seed.linkedWidgets = (seed.linkedWidgets ?? []).filter((widget) => widget !== native);
    mode = node.addWidget("combo", "种子控制", initialMode, () => {}, { values: SEED_MODES });
    const addedIndex = node.widgets.indexOf(mode);
    const seedIndex = node.widgets.indexOf(seed);
    if (addedIndex >= 0 && seedIndex >= 0) {
        node.widgets.splice(addedIndex, 1);
        node.widgets.splice(seedIndex + 1, 0, mode);
    }
}

function prepareSeedForQueue(node, graphData) {
    const seedWidget = getWidget(node, "随机种子");
    const modeWidget = getWidget(node, "种子控制");
    if (!seedWidget || !modeWidget) return;

    const current = Math.max(0, Math.min(SEED_MAX, Math.trunc(Number(seedWidget.value) || 0)));
    const mode = normalizeSeedMode(modeWidget.value);
    let queued = current;
    if (mode === "随机") queued = randomSeed(current);
    else if (mode === "增加") queued = current >= SEED_MAX ? 0 : current + 1;
    else if (mode === "减少") queued = current <= 0 ? SEED_MAX : current - 1;

    seedWidget.value = queued;
    modeWidget.value = mode;
    seedWidget.callback?.(queued);
    const promptNode = graphData?.output?.[String(node.id)] ?? graphData?.output?.[node.id];
    if (promptNode?.inputs) promptNode.inputs["随机种子"] = queued;
    const workflowNode = graphData?.workflow?.nodes?.find((item) => String(item?.id) === String(node.id));
    if (Array.isArray(workflowNode?.widgets_values)) {
        const index = node.widgets.indexOf(seedWidget);
        if (index >= 0) workflowNode.widgets_values[index] = queued;
    }
    node.setDirtyCanvas?.(true, true);
    console.info(`[AI蛮子] 本轮实际种子：${queued}（${mode}）`);
}

function refreshTemplateLayout(node) {
    // Keep a useful fixed preview viewport without allowing long content to grow
    // the whole node. The multiline widget scrolls internally.
    node.setSize([Math.max(520, Number(node.size?.[0]) || 520), 420]);
    const content = getWidget(node, "提交给模型的内容");
    if (content?.inputEl) {
        content.inputEl.style.minHeight = "240px";
        content.inputEl.style.maxHeight = "240px";
        content.inputEl.style.overflowY = "auto";
    }
    app.graph.setDirtyCanvas(true, true);
}

function syncInferenceUI(node) {
    // Workflows saved before the mmproj rename can retain the former widget client-side.
    // Rename it in place so old graphs do not show an orphaned control after reload.
    const legacy = getWidget(node, "多模态辅助模型");
    if (legacy && !getWidget(node, "mmproj")) legacy.name = "mmproj";
    if (legacy && getWidget(node, "mmproj") !== legacy) setWidgetVisible(node, "多模态辅助模型", false);
    const online = getWidget(node, "推理方式")?.value === "在线推理";
    ["在线_API_URL", "在线_API_Key", "在线_模型_ID"].forEach((name) => setWidgetVisible(node, name, online));
    ["启用_ninfer", "启用思考模式", "推理后卸载模型", "模型"].forEach(
        (name) => setWidgetVisible(node, name, !online),
    );
    const enabled = !!getWidget(node, "启用_ninfer")?.value;
    setWidgetVisible(node, "mmproj", !online && !enabled);
    const model = getWidget(node, "模型");
    if (model && Array.isArray(model.options?.values)) {
        if (!model.__allModels) model.__allModels = [...model.options.values];
        const filtered = model.__allModels.filter((value) => enabled ? value.toLowerCase().endsWith(".ninfer") : !value.toLowerCase().endsWith(".ninfer"));
        const actualModels = filtered.filter((value) => value !== "自动匹配");
        model.options.values = filtered.length ? filtered : ["未找到模型"];
        // Repair a workflow saved while the old mmproj/model widgets were misaligned.
        if (model.value === "自动匹配" && actualModels.length) model.value = actualModels[0];
        if (!model.options.values.includes(model.value)) model.value = actualModels[0] ?? model.options.values[0];
    }
    refreshLayout(node);
}

async function parseTemplateFile(node, value, showAlert = true) {
    const contentWidget = getWidget(node, "提交给模型的内容");
    if (contentWidget) contentWidget.value = "正在解析，请稍候……";
    node.setDirtyCanvas?.(true, true);
    try {
        const parsedResponse = await fetch("/aimanzi/parse-template", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file: value }),
        });
        const parsed = await parsedResponse.json();
        if (!parsedResponse.ok || !parsed.ok) throw new Error(parsed.error || `解析失败：HTTP ${parsedResponse.status}`);
        if (contentWidget) {
            contentWidget.value = parsed.content;
            contentWidget.callback?.(parsed.content);
        }
        const button = getWidget(node, "上传模板 / Skill");
        if (button) button.value = `已解析 ${parsed.characters} 字符`;
        node.graph?.setDirtyCanvas?.(true, true);
        refreshTemplateLayout(node);
        return true;
    } catch (error) {
        if (contentWidget) contentWidget.value = "";
        if (showAlert) alert(`模板/Skill 处理失败：${error?.message ?? error}`);
        node.setDirtyCanvas?.(true, true);
        return false;
    }
}

function uploadTemplate(node) {
    const chooser = document.createElement("input");
    chooser.type = "file";
    chooser.accept = ".txt,.md,.markdown,.skill,.zip,text/plain,text/markdown,application/zip";
    chooser.onchange = async () => {
        const file = chooser.files?.[0];
        if (!file) return;
        const extension = file.name.toLowerCase().split(".").pop();
        if (!["txt", "md", "markdown", "skill", "zip"].includes(extension)) {
            alert("仅支持 .txt、.md、.markdown、.skill 或包含 SKILL.md 的 .zip 文件。");
            return;
        }
        const data = new FormData();
        // ComfyUI's generic upload endpoint accepts arbitrary files and stores them in input/.
        data.append("image", file, file.name);
        data.append("type", "input");
        try {
            const response = await fetch("/upload/image", { method: "POST", body: data });
            if (!response.ok) throw new Error(`上传失败：HTTP ${response.status}`);
            const result = await response.json();
            const value = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
            const fileWidget = getWidget(node, "TXT/MD 文件");
            if (fileWidget) fileWidget.value = value;
            await parseTemplateFile(node, value);
        } catch (error) {
            const contentWidget = getWidget(node, "提交给模型的内容");
            if (contentWidget) contentWidget.value = "";
            alert(`模板/Skill 处理失败：${error?.message ?? error}`);
            node.setDirtyCanvas?.(true, true);
        }
    };
    chooser.click();
}

function orderMediaInputs(node) {
    const images = node.inputs.filter((input) => /^图像_\d+$/.test(input.name))
        .sort((a, b) => Number(a.name.slice(3)) - Number(b.name.slice(3)));
    const video = node.inputs.filter((input) => input.name === "视频");
    const rest = node.inputs.filter((input) => !/^图像_\d+$/.test(input.name) && input.name !== "视频");
    // Template stays first; independent images follow it; the IMAGE video batch stays last.
    node.inputs = [...rest, ...images, ...video];
}

function dynamicImageInputs(node) {
    const images = node.inputs.filter((input) => /^图像_\d+$/.test(input.name));
    const last = images.at(-1);
    if (images.length < MAX_IMAGE_INPUTS && (!last || last.link != null)) {
        const number = images.length + 1;
        node.addInput(`图像_${number}`, "IMAGE", { optional: true });
        orderMediaInputs(node);
        refreshLayout(node);
    }
}

app.registerExtension({
    name: "AIManzi.MultimodalPrompt",
    async beforeQueuePrompt(graphData) {
        for (const node of app.graph?._nodes ?? []) {
            if (node.comfyClass === NODE && node.mode !== 2 && node.mode !== 4) {
                prepareSeedForQueue(node, graphData);
            }
        }
        return graphData;
    },
    nodeCreated(node) {
        if (node.comfyClass === TEMPLATE_NODE) {
            if (!getWidget(node, "上传模板 / Skill")) {
                node.addWidget("button", "上传模板 / Skill", "选择文件", () => uploadTemplate(node));
            }
            // onConfigure runs after a saved workflow restores node.size. Hook it so
            // legacy giant nodes are compacted after that restored size is applied.
            const previousConfigure = node.onConfigure;
            node.onConfigure = function (info, ...args) {
                previousConfigure?.call(this, info, ...args);
                const saved = info?.widgets_values;
                const contentWidget = getWidget(this, "提交给模型的内容");
                // Older nodes saved [filename, upload-button-label]. Without this
                // migration the former button label becomes the new model content.
                const legacyTemplateValues = Array.isArray(saved) && (
                    saved.length === 1
                    || (saved.length === 2 && ["选择文件", "上传模板 / Skill"].includes(String(saved[1] ?? "")))
                );
                if (legacyTemplateValues && contentWidget) {
                    contentWidget.value = "";
                }
                requestAnimationFrame(() => {
                    refreshTemplateLayout(this);
                    const file = String(getWidget(this, "TXT/MD 文件")?.value ?? "").trim();
                    const content = String(getWidget(this, "提交给模型的内容")?.value ?? "").trim();
                    if (file && !content) parseTemplateFile(this, file, false);
                });
            };
            requestAnimationFrame(() => refreshTemplateLayout(node));
            return;
        }
        if (node.comfyClass !== NODE) return;
        ensureSeedControl(node);
        // v1.3 inserts strategy and seed directly below the prompt. Older workflows
        // stored only the former eleven backend widgets, so LiteGraph initially maps
        // those positional values onto the new fields. Restore them by their old names.
        const previousConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            previousConfigure?.call(this, info);
            const values = info?.widgets_values;
            // v1.4 workflows contain the removed 模板处理 value immediately
            // before 推理方式. Restore every surviving field by meaning rather
            // than allowing LiteGraph to shift the remaining controls.
            const removedModeIndex = Array.isArray(values)
                ? values.findIndex((value) => ["智能精简", "完整无损", "极速核心"].includes(value))
                : -1;
            if (removedModeIndex >= 3 && ["本地推理", "在线推理"].includes(values[removedModeIndex + 1])) {
                getWidget(this, "文字要求").value = values[0];
                getWidget(this, "推理策略").value = values[1];
                getWidget(this, "随机种子").value = values[2];
                if (removedModeIndex === 4) getWidget(this, "种子控制").value = normalizeSeedMode(values[3]);
                const remainingNames = [
                    "推理方式", "在线_API_URL", "在线_API_Key", "在线_模型_ID", "启用_ninfer",
                    "启用思考模式", "推理后卸载模型", "视频分析精度", "模型", "mmproj",
                ];
                remainingNames.forEach((name, index) => {
                    const widget = getWidget(this, name);
                    if (widget) widget.value = values[removedModeIndex + 1 + index];
                });
            }
            // v1.4 adds 模板处理 after seed. Restore v1.3's thirteen positional
            // values by name before LiteGraph can shift 推理方式 and all later fields.
            if (Array.isArray(values) && values.length === 13 && ["本地推理", "在线推理"].includes(values[3])) {
                const v13Names = [
                    "文字要求", "推理策略", "随机种子", "推理方式", "在线_API_URL", "在线_API_Key",
                    "在线_模型_ID", "启用_ninfer", "启用思考模式", "推理后卸载模型", "视频分析精度", "模型", "mmproj",
                ];
                v13Names.forEach((name, index) => {
                    const widget = getWidget(this, name);
                    if (widget) widget.value = values[index];
                });
            }
            if (Array.isArray(values) && values.length === 11 && ["本地推理", "在线推理"].includes(values[1])) {
                const oldNames = [
                    "文字要求", "推理方式", "在线_API_URL", "在线_API_Key", "在线_模型_ID",
                    "启用_ninfer", "启用思考模式", "推理后卸载模型", "视频分析精度", "模型", "mmproj",
                ];
                oldNames.forEach((name, index) => {
                    const widget = getWidget(this, name);
                    if (widget) widget.value = values[index];
                });
                const strategy = getWidget(this, "推理策略");
                const seed = getWidget(this, "随机种子");
                if (strategy) strategy.value = "普通推理";
                if (seed) seed.value = 0;
            }
            queueMicrotask(() => {
                ensureSeedControl(this);
                syncInferenceUI(this);
            });
        };
        // Migrate the v1.2 IMAGE socket name without breaking its connection.
        const formerFrameInput = node.inputs?.find((input) => input.name === "视频帧输入");
        if (formerFrameInput) {
            formerFrameInput.name = "视频";
            formerFrameInput.type = "IMAGE";
        }
        // v1.2 removes the former VIDEO socket. Old workflows are migrated by
        // deleting that stale socket; users reconnect the loader's IMAGE batch.
        const legacyVideoIndex = node.inputs?.findIndex((input) => input.name === "视频输入") ?? -1;
        if (legacyVideoIndex >= 0) node.removeInput(legacyVideoIndex);
        const inferenceMode = getWidget(node, "推理方式");
        if (inferenceMode) {
            const previousMode = inferenceMode.callback;
            inferenceMode.callback = (...args) => {
                previousMode?.(...args);
                queueMicrotask(() => syncInferenceUI(node));
            };
        }
        const ninfer = getWidget(node, "启用_ninfer");
        if (ninfer) {
            const previous = ninfer.callback;
            ninfer.callback = (...args) => {
                previous?.(...args);
                queueMicrotask(() => syncInferenceUI(node));
            };
        }
        const prior = node.onConnectionsChange;
        node.onConnectionsChange = (...args) => {
            prior?.apply(node, args);
            queueMicrotask(() => dynamicImageInputs(node));
        };
        syncInferenceUI(node);
        dynamicImageInputs(node);
        orderMediaInputs(node);
        requestAnimationFrame(() => refreshLayout(node));
    },
});
