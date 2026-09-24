import { app } from "/scripts/app.js";

const NODE = "AIManziMultimodalPrompt";
const MAX_IMAGE_INPUTS = 10;

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

function syncNInferUI(node) {
    // Workflows saved before the mmproj rename can retain the former widget client-side.
    // Rename it in place so old graphs do not show an orphaned control after reload.
    const legacy = getWidget(node, "多模态辅助模型");
    if (legacy && !getWidget(node, "mmproj")) legacy.name = "mmproj";
    if (legacy && getWidget(node, "mmproj") !== legacy) setWidgetVisible(node, "多模态辅助模型", false);
    const enabled = !!getWidget(node, "启用_ninfer")?.value;
    setWidgetVisible(node, "mmproj", !enabled);
    const model = getWidget(node, "模型");
    if (!model || !Array.isArray(model.options?.values)) return;
    if (!model.__allModels) model.__allModels = [...model.options.values];
    const filtered = model.__allModels.filter((value) => enabled ? value.toLowerCase().endsWith(".ninfer") : !value.toLowerCase().endsWith(".ninfer"));
    const actualModels = filtered.filter((value) => value !== "自动匹配");
    model.options.values = filtered.length ? filtered : ["未找到模型"];
    // Repair a workflow saved while the old mmproj/model widgets were misaligned.
    if (model.value === "自动匹配" && actualModels.length) model.value = actualModels[0];
    if (!model.options.values.includes(model.value)) model.value = actualModels[0] ?? model.options.values[0];
    refreshLayout(node);
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
        const response = await fetch("/upload/image", { method: "POST", body: data });
        if (!response.ok) {
            throw new Error(`上传失败：HTTP ${response.status}`);
        }
        const result = await response.json();
        const value = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
        const fileWidget = getWidget(node, "TXT/MD 文件");
        if (fileWidget) {
            fileWidget.value = value;
            node.setDirtyCanvas?.(true, true);
        }
    };
    chooser.click();
}

function orderMediaInputs(node) {
    const images = node.inputs.filter((input) => /^图像_\d+$/.test(input.name))
        .sort((a, b) => Number(a.name.slice(3)) - Number(b.name.slice(3)));
    const video = node.inputs.filter((input) => input.name === "视频输入");
    const rest = node.inputs.filter((input) => !/^图像_\d+$/.test(input.name) && input.name !== "视频输入");
    // Template stays first; every image follows it; VIDEO is always below the final image socket.
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
    nodeCreated(node) {
        if (node.comfyClass === "AIManziReadText") {
            if (!getWidget(node, "上传模板 / Skill")) {
                node.addWidget("button", "上传模板 / Skill", "选择文件", () => uploadTemplate(node));
                requestAnimationFrame(() => refreshLayout(node));
            }
            return;
        }
        if (node.comfyClass !== NODE) return;
        const ninfer = getWidget(node, "启用_ninfer");
        if (ninfer) {
            const previous = ninfer.callback;
            ninfer.callback = (...args) => {
                previous?.(...args);
                queueMicrotask(() => syncNInferUI(node));
            };
        }
        const prior = node.onConnectionsChange;
        node.onConnectionsChange = (...args) => {
            prior?.apply(node, args);
            queueMicrotask(() => dynamicImageInputs(node));
        };
        syncNInferUI(node);
        dynamicImageInputs(node);
        orderMediaInputs(node);
        requestAnimationFrame(() => refreshLayout(node));
    },
});
