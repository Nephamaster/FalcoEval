import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Checkbox,
  Collapse,
  Descriptions,
  Divider,
  Empty,
  Form,
  Input,
  Layout,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography
} from "antd";
import {
  ArrowRightOutlined,
  DatabaseOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined
} from "@ant-design/icons";
import {
  fetchDatasetSelection,
  fetchEvaluation,
  fetchMeta,
  registerModel,
  startEvaluation
} from "./api";
import type { EvaluationJob, MetaResponse } from "./types";

const { Content } = Layout;
const { Paragraph, Text, Title } = Typography;

type EvaluateFormValues = {
  modelName?: string;
  customModel?: string;
  datasetNames: string[];
  selectedMetrics: string[];
  tpSizeChoice?: string;
  tpSizeInput?: string;
};

function prettyJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function stripDatasetOverviewTitle(markdown: string) {
  return markdown.replace(/^##\s*数据集概览\s*/u, "").trim();
}

function renderSimpleMarkdown(markdown: string) {
  const lines = stripDatasetOverviewTitle(markdown).split(/\r?\n/);
  const elements: JSX.Element[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (!listItems.length) return;
    elements.push(
      <ul key={`list-${elements.length}`}>
        {listItems.map((item, index) => (
          <li key={`${item}-${index}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>
    );
    listItems = [];
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      return;
    }
    if (line.startsWith("####")) {
      flushList();
      elements.push(<h5 key={index}>{renderInlineMarkdown(line.replace(/^####\s*/, ""))}</h5>);
      return;
    }
    if (line.startsWith("###")) {
      flushList();
      elements.push(<h4 key={index}>{renderInlineMarkdown(line.replace(/^###\s*/, ""))}</h4>);
      return;
    }
    if (line.startsWith("##")) {
      flushList();
      elements.push(<h3 key={index}>{renderInlineMarkdown(line.replace(/^##\s*/, ""))}</h3>);
      return;
    }
    if (/^[-*]\s+/.test(line)) {
      listItems.push(line.replace(/^[-*]\s+/, ""));
      return;
    }
    flushList();
    elements.push(<p key={index}>{renderInlineMarkdown(line)}</p>);
  });

  flushList();
  return elements;
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function datasetResultRows(result: Record<string, unknown> | null | undefined) {
  const datasets = result?.datasets;
  if (!datasets || typeof datasets !== "object") {
    return [];
  }

  return Object.entries(datasets as Record<string, any>).map(([dataset, value]) => ({
    key: dataset,
    dataset,
    dataType: value.data_type,
    metrics: Array.isArray(value.metrics_used) ? value.metrics_used.join(", ") : "",
    selectedMetrics: Array.isArray(value.metrics_used) ? value.metrics_used : [],
    result: value.result ?? {}
  }));
}

function formatMetricValue(value: unknown): string | number {
  if (typeof value === "number") {
    return Number(value.toFixed(4));
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${key}: ${formatMetricValue(item)}`)
      .join("，");
  }
  return String(value ?? "");
}

function metricLabel(metric: string) {
  const labels: Record<string, string> = {
    Accuracy: "准确率",
    TokenEfficiency: "Token 效率",
    EM: "精确匹配",
    F1: "F1",
    BLEU: "BLEU",
    ROUGE: "ROUGE",
    Precision: "精确率",
    "Latency mean (ms)": "平均延迟（ms）",
    "Cold start load (ms)": "冷启动时间（ms）"
  };
  return labels[metric] ?? metric;
}

function metricRows(metricResult: Record<string, unknown>, selectedMetrics: string[]) {
  const visibleMetrics = [...selectedMetrics, "Latency mean (ms)", "Cold start load (ms)"];
  return visibleMetrics
    .filter((metric) => Object.prototype.hasOwnProperty.call(metricResult, metric))
    .map((metric) => ({
      key: metric,
      metric: metricLabel(metric),
      value: formatMetricValue(metricResult[metric])
    }));
}

function downloadJson(filename: string, value: unknown) {
  const blob = new Blob([prettyJson(value)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function safeFilenamePart(value: string) {
  return value.replace(/[\\/:*?"<>|\s]+/g, "_").replace(/^_+|_+$/g, "");
}

function evaluationDownloadFilename(result: Record<string, unknown>, fallbackJobId: string) {
  const model = typeof result.model === "string" ? result.model : "model";
  const datasets = result.datasets && typeof result.datasets === "object"
    ? Object.keys(result.datasets as Record<string, unknown>).join("+")
    : "datasets";
  const date = new Date().toISOString().slice(0, 10);
  const filename = `${safeFilenamePart(model)}_${safeFilenamePart(datasets)}_${date}.json`;
  return filename === "__.json" ? `falcoeval_${fallbackJobId}.json` : filename;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function App() {
  const { message } = AntApp.useApp();
  const [form] = Form.useForm<EvaluateFormValues>();
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [datasetInfo, setDatasetInfo] = useState("");
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [registering, setRegistering] = useState(false);
  const [registerAlias, setRegisterAlias] = useState("");
  const [registerPath, setRegisterPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<EvaluationJob | null>(null);

  const selectedDatasetNames = Form.useWatch("datasetNames", form) ?? [];
  const running = job?.status === "queued" || job?.status === "running";

  async function loadMeta() {
    setLoadingMeta(true);
    setMetaError(null);
    try {
      const nextMeta = await fetchMeta();
      setMeta(nextMeta);
      setDatasetInfo(nextMeta.default_dataset_info);
      setAvailableMetrics(nextMeta.default_metrics);
      form.setFieldsValue({
        modelName: nextMeta.models[0]?.value,
        datasetNames: nextMeta.default_datasets,
        selectedMetrics: [],
        tpSizeChoice: nextMeta.default_gpu_choice
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "元数据加载失败";
      setMetaError(errorMessage);
      message.error(errorMessage);
    } finally {
      setLoadingMeta(false);
    }
  }

  useEffect(() => {
    void loadMeta();
  }, []);

  useEffect(() => {
    if (!selectedDatasetNames.length) {
      setDatasetInfo("## 数据集概览\n请选择至少一个数据集。");
      setAvailableMetrics([]);
      form.setFieldValue("selectedMetrics", []);
      return;
    }

    let disposed = false;
    fetchDatasetSelection(selectedDatasetNames)
      .then((selection) => {
        if (disposed) return;
        const currentMetrics = form.getFieldValue("selectedMetrics") ?? [];
        setDatasetInfo(selection.dataset_info);
        setAvailableMetrics(selection.metrics);
        form.setFieldValue(
          "selectedMetrics",
          currentMetrics.filter((metric: string) => selection.metrics.includes(metric))
        );
      })
      .catch((error) => {
        if (!disposed) {
          message.error(error instanceof Error ? error.message : "数据集信息加载失败");
        }
      });

    return () => {
      disposed = true;
    };
  }, [selectedDatasetNames.join("|")]);

  useEffect(() => {
    if (!job || !running) return;

    const timer = window.setInterval(() => {
      fetchEvaluation(job.id)
        .then(setJob)
        .catch((error) => {
          message.error(error instanceof Error ? error.message : "评测状态更新失败");
          window.clearInterval(timer);
        });
    }, 2000);

    return () => window.clearInterval(timer);
  }, [job?.id, running]);

  const datasetSummary = useMemo(() => {
    if (!meta) return [];
    return selectedDatasetNames
      .map((name) => meta.datasets.find((dataset) => dataset.name === name))
      .filter(Boolean);
  }, [meta, selectedDatasetNames]);

  const [route, setRoute] = useState(window.location.pathname);

  useEffect(() => {
    const handlePopState = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigate(path: string) {
    window.history.pushState({}, "", path);
    setRoute(path);
  }

  async function handleRegister(values: { alias: string; modelPath: string }) {
    setRegistering(true);
    try {
      const response = await registerModel(values.alias, values.modelPath);
      setMeta((current) => (current ? { ...current, models: response.models } : current));
      form.setFieldValue("modelName", response.selected);
      setRegisterAlias("");
      setRegisterPath("");
      message.success(response.status);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "模型注册失败");
    } finally {
      setRegistering(false);
    }
  }

  async function handleEvaluate(values: EvaluateFormValues) {
    setSubmitting(true);
    setJob(null);
    try {
      const nextJob = await startEvaluation({
        model_name: values.modelName,
        custom_model: values.customModel,
        dataset_names: values.datasetNames,
        selected_metrics: values.selectedMetrics ?? [],
        tp_size_choice: values.tpSizeChoice,
        tp_size_input: values.tpSizeInput
      });
      setJob(nextJob);
      message.info("评测任务已提交");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "评测提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AntApp>
      <Layout className="app-shell">
        <Content className="workspace">
          {route === "/" ? (
            <div className="home-page">
              <nav className="home-nav">
                <button className="brand-button" type="button" onClick={() => navigate("/")}>
                  FalcoEval
                </button>
              </nav>

              <header className="home-hero">
                <div className="home-copy">
                  <h1 className="home-title">LLM Evaluation & Benchmarking</h1>
                  <Paragraph className="home-description">
                    <span>大语言模型自动化评测平台，集成模型推理、数据集基准、指标计算与运行性能统计</span>
                    <span>让研究和工程团队用一致流程完成可复现的模型评估</span>
                  </Paragraph>
                  <Space className="home-actions" size={14} wrap>
                    <Button type="primary" size="large" onClick={() => navigate("/evaluate")}>
                      进入评测控制台
                      <ArrowRightOutlined className="arrow-up-right" />
                    </Button>
                  </Space>

                  {metaError ? (
                    <Alert
                      className="home-meta-error"
                      type="warning"
                      showIcon
                      message="平台配置加载失败"
                      description={metaError}
                    />
                  ) : null}

                  <div className="home-metrics">
                    <div>
                      <strong>{meta?.datasets.length ?? 0}</strong>
                      <span>Benchmarks</span>
                    </div>
                    <div>
                      <strong>{meta?.models.length ?? 0}</strong>
                      <span>Registered Models</span>
                    </div>
                    <div>
                      <strong>{meta?.available_gpu_count ?? 0}</strong>
                      <span>Visible GPUs</span>
                    </div>
                  </div>
                </div>
              </header>

              <section className="home-intro">
                <div>
                  <Title className="feature-title" level={3}>统一基准，一键评测</Title>
                  <Paragraph className="feature-copy">
                    覆盖多类主流数据集与指标。选择模型和基准后即可发起评测，减少脚本切换和结果整理成本。
                  </Paragraph>
                </div>
                <div>
                  <Title className="feature-title" level={3}>结果清晰，便于对比</Title>
                  <Paragraph className="feature-copy">
                    统一汇总指标得分、模型表现与运行效率，便于模型选型、版本回归和实验记录。
                  </Paragraph>
                </div>
              </section>
            </div>
          ) : (
            <>
              <header className="topbar">
                <div>
                  <Title className="console-title" level={2}>FalcoEval Console</Title>
                  <Paragraph className="console-subtitle">大语言模型自动化评测控制台</Paragraph>
                </div>
                <Space className="console-nav-actions" wrap>
                  <Button className="ghost-pill-button" onClick={() => navigate("/")}>返回首页</Button>
                  <Button icon={<ReloadOutlined />} onClick={loadMeta} loading={loadingMeta}>
                    刷新配置
                  </Button>
                </Space>
              </header>

              {loadingMeta && !meta ? (
                <div className="loading">
                  <Spin size="large" />
                </div>
              ) : (
                <>
                  {metaError ? (
                    <Alert
                      className="console-meta-error"
                      type="warning"
                      showIcon
                      message="未能加载默认模型和 benchmark"
                      description={metaError}
                    />
                  ) : null}
                <div className="main-grid">
              <aside className="side-panel">
                <Card title="运行状态" className="compact-card">
                  <Space direction="vertical" size={12} className="full-width">
                    <Statistic title="可见 GPU" value={meta?.available_gpu_count ?? 0} />
                    <div>
                      <Text type="secondary">后端</Text>
                      <div>
                        <Tag color="blue">{meta?.backend ?? "sglang"}</Tag>
                      </div>
                    </div>
                    <Alert
                      type="info"
                      showIcon
                      message="同一模型和 tp_size 会复用已加载引擎。"
                    />
                  </Space>
                </Card>

                <Card title="数据集概览" className="compact-card">
                  <div className="dataset-info">{renderSimpleMarkdown(datasetInfo)}</div>
                </Card>
              </aside>

              <main className="content-panel">
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleEvaluate}
                  initialValues={{ selectedMetrics: [] }}
                >
                  <Card title="模型配置">
                    <div className="form-grid">
                      <Form.Item label="已注册模型" name="modelName">
                        <Select
                          allowClear
                          placeholder="选择已注册模型"
                          options={meta?.models.map((model) => ({
                            label: model.label,
                            value: model.value
                          }))}
                        />
                      </Form.Item>
                      <Form.Item label="临时模型路径" name="customModel">
                        <Input placeholder="可选：本地路径或 HuggingFace 模型路径" />
                      </Form.Item>
                    </div>

                    <Collapse
                      ghost
                      items={[
                        {
                          key: "register",
                          label: "注册模型",
                          children: (
                            <div className="form-grid with-action">
                              <div>
                                <Text className="field-label">模型别名</Text>
                                <Input
                                  value={registerAlias}
                                  onChange={(event) => setRegisterAlias(event.target.value)}
                                  placeholder="例如：Qwen2.5-7B-Instruct"
                                />
                              </div>
                              <div>
                                <Text className="field-label">模型路径</Text>
                                <Input
                                  value={registerPath}
                                  onChange={(event) => setRegisterPath(event.target.value)}
                                  placeholder="例如：Qwen/Qwen2.5-7B-Instruct"
                                />
                              </div>
                              <Button
                                type="primary"
                                icon={<PlusOutlined />}
                                loading={registering}
                                onClick={() => handleRegister({ alias: registerAlias, modelPath: registerPath })}
                              >
                                注册
                              </Button>
                            </div>
                          )
                        }
                      ]}
                    />
                  </Card>

                  <Card title="显卡配置">
                    <div className="form-grid">
                      <Form.Item label="显卡数量（下拉选择）" name="tpSizeChoice">
                        <Select options={meta?.gpu_choices.map((item) => ({ label: item, value: item }))} />
                      </Form.Item>
                      <Form.Item label="显卡数量（手动覆盖）" name="tpSizeInput">
                        <Input placeholder="可选：输入 1 / 2 / 4 等正整数" />
                      </Form.Item>
                    </div>
                  </Card>

                  <Card title="数据集与指标">
                    <Form.Item
                      label="评测基准"
                      name="datasetNames"
                      rules={[{ required: true, message: "请至少选择一个数据集" }]}
                    >
                      <Select
                        mode="multiple"
                        placeholder="选择评测基准"
                        options={meta?.datasets.map((dataset) => ({
                          label: dataset.name,
                          value: dataset.name
                        }))}
                      />
                    </Form.Item>

                    <Form.Item label="评测指标" name="selectedMetrics">
                      <Checkbox.Group options={availableMetrics} />
                    </Form.Item>

                    <div className="dataset-tags">
                      {datasetSummary.map((dataset) =>
                        dataset ? (
                          <Tag icon={<DatabaseOutlined />} key={dataset.name}>
                            {dataset.name} · {dataset.data_type}
                          </Tag>
                        ) : null
                      )}
                    </div>
                  </Card>

                  <div className="actions">
                    <Button
                      type="primary"
                      size="large"
                      icon={<PlayCircleOutlined />}
                      htmlType="submit"
                      loading={submitting || running}
                    >
                      开始评测
                    </Button>
                  </div>
                </Form>

                <Card title="评测结果">
                  {!job ? (
                    <Empty description="暂无评测结果" />
                  ) : (
                    <Space direction="vertical" size={16} className="full-width">
                      <Descriptions size="small" column={3} bordered>
                        <Descriptions.Item label="状态">
                          <Tag
                            color={
                              job.status === "succeeded"
                                ? "green"
                                : job.status === "failed"
                                  ? "red"
                                  : "gold"
                            }
                          >
                            {job.status}
                          </Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="更新时间">{formatDateTime(job.updated_at)}</Descriptions.Item>
                      </Descriptions>
                      {running ? <Alert type="warning" showIcon message="评测运行中，请等待结果刷新。" /> : null}
                      {job.error ? <Alert type="error" showIcon message={job.error} /> : null}
                      {job.result ? (
                        <>
                          <Table
                            className="result-table"
                            size="small"
                            pagination={false}
                            dataSource={datasetResultRows(job.result)}
                            expandable={{
                              expandedRowRender: (record) => (
                                <Table
                                  size="small"
                                  pagination={false}
                                  dataSource={metricRows(record.result, record.selectedMetrics)}
                                  columns={[
                                    { title: "指标", dataIndex: "metric", key: "metric" },
                                    { title: "数值", dataIndex: "value", key: "value" }
                                  ]}
                                />
                              )
                            }}
                            columns={[
                              { title: "数据集", dataIndex: "dataset", key: "dataset" },
                              { title: "任务类型", dataIndex: "dataType", key: "dataType" },
                              { title: "评测指标", dataIndex: "metrics", key: "metrics" }
                            ]}
                          />
                          <div className="result-actions">
                            <Button onClick={() => downloadJson(evaluationDownloadFilename(job.result!, job.id), job.result)}>
                              下载 JSON
                            </Button>
                          </div>
                        </>
                      ) : null}
                    </Space>
                  )}
                </Card>
              </main>
                </div>
                </>
              )}
            </>
          )}
        </Content>
      </Layout>
    </AntApp>
  );
}

export default App;
