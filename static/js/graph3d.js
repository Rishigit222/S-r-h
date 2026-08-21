/* Interactive Force-Directed Knowledge Graph Visualizer for s@r@h */

class KnowledgeGraph3D {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.nodes = new Map();
    this.edges = [];
    this.selectedNode = null;
    this.draggedNode = null;

    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.initEvents();
    this.loadGraphData();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width || 800;
    this.canvas.height = 580;
  }

  async loadGraphData() {
    try {
      const resp = await fetch("/graph/triples?limit=80");
      const data = await resp.json();
      const triples = data.triples || [];

      this.nodes.clear();
      this.edges = [];

      triples.forEach((t) => {
        if (!this.nodes.has(t.subject)) {
          this.nodes.set(t.subject, {
            id: t.subject,
            x: this.canvas.width / 2 + (Math.random() - 0.5) * 350,
            y: this.canvas.height / 2 + (Math.random() - 0.5) * 350,
            vx: 0,
            vy: 0,
            radius: 14,
            color: "#00f0ff",
          });
        }

        if (!this.nodes.has(t.object)) {
          this.nodes.set(t.object, {
            id: t.object,
            x: this.canvas.width / 2 + (Math.random() - 0.5) * 350,
            y: this.canvas.height / 2 + (Math.random() - 0.5) * 350,
            vx: 0,
            vy: 0,
            radius: 11,
            color: "#a855f7",
          });
        }

        this.edges.push({
          source: this.nodes.get(t.subject),
          target: this.nodes.get(t.object),
          predicate: t.predicate,
        });
      });

      this.runPhysicsLoop();
    } catch (e) {
      console.warn("Could not load knowledge graph triples:", e);
    }
  }

  initEvents() {
    this.canvas.addEventListener("mousedown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      for (let node of this.nodes.values()) {
        const dx = node.x - x;
        const dy = node.y - y;
        if (Math.sqrt(dx * dx + dy * dy) < node.radius + 5) {
          this.draggedNode = node;
          this.selectedNode = node;
          this.updateInspector(node);
          break;
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (this.draggedNode) {
        const rect = this.canvas.getBoundingClientRect();
        this.draggedNode.x = e.clientX - rect.left;
        this.draggedNode.y = e.clientY - rect.top;
      }
    });

    window.addEventListener("mouseup", () => {
      this.draggedNode = null;
    });
  }

  updateInspector(node) {
    const el = document.getElementById("graph-node-details");
    if (!el) return;
    const connected = this.edges.filter(
      (e) => e.source.id === node.id || e.target.id === node.id
    );

    el.innerHTML = `
      <h4 style="color: var(--accent-cyan); margin-bottom: 8px;">🔹 ${node.id}</h4>
      <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">Connected Relations: <strong>${connected.length}</strong></p>
      <div style="display: flex; flex-direction: column; gap: 6px;">
        ${connected.map((e) => `
          <div style="font-size: 12px; padding: 6px 10px; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid var(--accent-violet);">
            (${e.source.id}) ➔ <span style="color: var(--accent-cyan)">[${e.predicate}]</span> ➔ (${e.target.id})
          </div>
        `).join("")}
      </div>
    `;
  }

  runPhysicsLoop() {
    const step = () => {
      // 1. Repulsion between all nodes
      const nodeList = Array.from(this.nodes.values());
      for (let i = 0; i < nodeList.length; i++) {
        for (let j = i + 1; j < nodeList.length; j++) {
          let n1 = nodeList[i];
          let n2 = nodeList[j];
          let dx = n2.x - n1.x;
          let dy = n2.y - n1.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 180) {
            let force = (180 - dist) / 180 * 1.5;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            if (n1 !== this.draggedNode) { n1.x -= fx; n1.y -= fy; }
            if (n2 !== this.draggedNode) { n2.x += fx; n2.y += fy; }
          }
        }
      }

      // 2. Spring attraction along edges
      this.edges.forEach((e) => {
        let dx = e.target.x - e.source.x;
        let dy = e.target.y - e.source.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let targetDist = 90;
        let force = (dist - targetDist) * 0.03;
        let fx = (dx / dist) * force;
        let fy = (dy / dist) * force;

        if (e.source !== this.draggedNode) { e.source.x += fx; e.source.y += fy; }
        if (e.target !== this.draggedNode) { e.target.x -= fx; e.target.y -= fy; }
      });

      // 3. Render
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      // Draw Edges
      this.edges.forEach((e) => {
        this.ctx.beginPath();
        this.ctx.strokeStyle = "rgba(168, 85, 247, 0.35)";
        this.ctx.lineWidth = 1.2;
        this.ctx.moveTo(e.source.x, e.source.y);
        this.ctx.lineTo(e.target.x, e.target.y);
        this.ctx.stroke();

        // Edge label
        const midX = (e.source.x + e.target.x) / 2;
        const midY = (e.source.y + e.target.y) / 2;
        this.ctx.font = "9px Inter, sans-serif";
        this.ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
        this.ctx.fillText(e.predicate, midX - 10, midY - 3);
      });

      // Draw Nodes
      nodeList.forEach((n) => {
        this.ctx.beginPath();
        this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        this.ctx.fillStyle = n.color;
        this.ctx.shadowColor = n.color;
        this.ctx.shadowBlur = n === this.selectedNode ? 20 : 8;
        this.ctx.fill();
        this.ctx.shadowBlur = 0;

        // Label
        this.ctx.font = "11px Inter, sans-serif";
        this.ctx.fillStyle = "#f8fafc";
        this.ctx.fillText(n.id, n.x + n.radius + 5, n.y + 4);
      });

      requestAnimationFrame(step);
    };

    step();
  }
}
