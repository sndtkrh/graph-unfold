struct Node {
    pos: vec2<f32>,
    vel: vec2<f32>,
    acc: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> node_src: array<Node>;
@group(0) @binding(1) var<storage, read_write> node_target: array<Node>;
@group(0) @binding(2) var<storage, read> adj_mat: array<f32>;

@compute
@workgroup_size(64)
fn compute_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let node_n = arrayLength(&node_src);
    let idx = global_invocation_id.x;

    let cpos = node_src[idx].pos;
    let cvel = node_src[idx].vel;
    let cacc = node_src[idx].acc;

    let dt = 0.05;
    let q = 0.2;
    let m = 1000.0;
    let k = 100.0;
    let mu = m * 0.4 * min(length(cvel), 0.2);
    let max_vel = 3.0;

    let pos: vec2<f32> = cpos + cvel * dt;
    
    var vel: vec2<f32> = cvel + cacc * dt;
    if vel.x > max_vel {
        vel.x = max_vel;
    }
    if vel.x < - max_vel {
        vel.x = - max_vel;
    }
    if vel.y > max_vel {
        vel.y = max_vel;
    }
    if vel.y < - max_vel {
        vel.y = - max_vel;
    }
    
    var acc: vec2<f32> = vec2<f32>(0.0, 0.0);

    var fF: vec2<f32> = vec2<f32>(0.0, 0.0); // friction
    if (length(cvel) > 0.000001) {
        fF = - normalize(cvel) * mu;
    }
    acc += fF / m;

    for(var i : u32 = 0u; i < node_n; i++) {
        if(idx == i) {
           continue; 
        }
        let dist = distance(cpos, node_src[i].pos);
        let dir: vec2<f32> = normalize(cpos - node_src[i].pos); // normalised direction
        let fQ: vec2<f32> = dir * (q * q) / (dist * dist);
        let fS: vec2<f32> = - dir * k * adj_mat[idx * node_n + i] * dist; // spring
        acc += (fQ + fS) / m; // ma = F
    }
    node_target[idx] = Node(pos, vel, acc);
}
