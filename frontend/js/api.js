/* Thin wrapper around the backend. Every call funnels through one place so
   error handling and the base URL are defined once. */

const API = (() => {
  const base = "";

  async function request(path, options = {}) {
    let response;
    try {
      response = await fetch(base + path, options);
    } catch (networkError) {
      throw new Error("The server did not respond. Check that it is running.");
    }

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}.`;
      try {
        const body = await response.json();
        if (body.detail) detail = body.detail;
      } catch (_) {
        /* the body was not JSON; keep the status message */
      }
      throw new Error(detail);
    }
    return response.json();
  }

  return {
    health: () => request("/api/health"),
    config: () => request("/api/config"),
    metrics: () => request("/api/metrics"),
    samples: () => request("/api/samples"),

    predict(file, { stages = true, heatmap = true } = {}) {
      const form = new FormData();
      form.append("file", file);
      form.append("show_stages", String(stages));
      form.append("show_heatmap", String(heatmap));
      return request("/api/predict", { method: "POST", body: form });
    },

    predictSample(name, { stages = true, heatmap = true } = {}) {
      const form = new FormData();
      form.append("name", name);
      form.append("show_stages", String(stages));
      form.append("show_heatmap", String(heatmap));
      return request("/api/predict/sample", { method: "POST", body: form });
    },
  };
})();
