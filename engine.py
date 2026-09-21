import pybullet as p
import random
import math
from config import *

all_pellets = {}
next_pellet_id = 1

def create_bounding_box():
    thickness = 1.0
    
    # Depth walls (Y-axis) to keep pellets within the camera's focus range
    colBoxIdY = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BOX_SIZE*3, thickness/2, BOX_SIZE*3]) # Made X-extent huge just in case they bounce on the corners, and Z-extent huge
    visBoxIdY = p.createVisualShape(p.GEOM_BOX, halfExtents=[BOX_SIZE*3, thickness/2, BOX_SIZE*3], rgbaColor=[1,1,1,0.0])
    
    # Near face
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxIdY, baseVisualShapeIndex=visBoxIdY, basePosition=[0, -BOX_SIZE/2 - thickness/2, BOX_SIZE/2])
    # Far face
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxIdY, baseVisualShapeIndex=visBoxIdY, basePosition=[0, BOX_SIZE/2 + thickness/2, BOX_SIZE/2])

def spawn_pellet(is_agglomerate=False):
    global next_pellet_id
    num_pellets = 1
    if is_agglomerate:
        num_pellets = random.randint(2, MAX_PELLETS_PER_AGGLOMERATE)
        
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=BASE_RADIUS)

    # initial postion of pellets.
    base_pos = [random.uniform(-BOX_SIZE*0.8, BOX_SIZE*0.8), 
                random.uniform(-BOX_SIZE/2 + BASE_RADIUS*3, BOX_SIZE/2 - BASE_RADIUS*3), 
                random.uniform(SPAWN_Z, SPAWN_Z + 4.0)]
                
    base_id = p.createMultiBody(baseMass=1.0, 
                                baseCollisionShapeIndex=colSphereId, 
                                basePosition=base_pos)
    
    bodies = [base_id]
    
    if is_agglomerate:
        occupied = [[0,0,0]]
        for i in range(num_pellets - 1):
            parent_idx = random.randint(0, len(occupied)-1)
            parent_pos = occupied[parent_idx]
            
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, math.pi)
            dist = random.uniform(1.6, 1.9) * BASE_RADIUS
            
            dx = dist * math.sin(phi) * math.cos(theta)
            dy = dist * math.sin(phi) * math.sin(theta)
            dz = dist * math.cos(phi)
            
            new_pos_rel = [parent_pos[0] + dx, parent_pos[1] + dy, parent_pos[2] + dz]
            occupied.append(new_pos_rel)
            
            world_pos = [base_pos[0] + new_pos_rel[0], base_pos[1] + new_pos_rel[1], base_pos[2] + new_pos_rel[2]]
            
            new_body = p.createMultiBody(baseMass=1.0, 
                                         baseCollisionShapeIndex=colSphereId, 
                                         basePosition=world_pos)
            bodies.append(new_body)
            
            # Disable collision between connected bodies
            p.setCollisionFilterPair(base_id, new_body, -1, -1, enableCollision=0)
            for other_b in bodies[:-1]:
                p.setCollisionFilterPair(other_b, new_body, -1, -1, enableCollision=0)
            
            p.createConstraint(parentBodyUniqueId=base_id,
                               parentLinkIndex=-1,
                               childBodyUniqueId=new_body,
                               childLinkIndex=-1,
                               jointType=p.JOINT_FIXED,
                               jointAxis=[0,0,0],
                               parentFramePosition=new_pos_rel,
                               childFramePosition=[0,0,0])

    for b in bodies:
        p.changeDynamics(b, -1, restitution=RESTITUTION, linearDamping=0.1, angularDamping=0.1)
        
    # Larger angle left/right velocity for realistic spread across the huge box
    vx = random.uniform(-6, 6)
    vy = random.uniform(-6, 6)
    vz = random.uniform(-2, -8)
    p.resetBaseVelocity(base_id, 
                        linearVelocity=[vx, vy, vz],
                        angularVelocity=[random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
                        
    pellet_id = next_pellet_id
    next_pellet_id += 1
    
    # Create debug text
    text_color = [1, 0.2, 0.2] if is_agglomerate else [0.2, 1, 0.2]
    text_id = p.addUserDebugText(str(pellet_id), base_pos, textColorRGB=text_color, textSize=1.5)
    
    all_pellets[base_id] = {
        'id': pellet_id,
        'is_agglomerate': is_agglomerate,
        'text_id': text_id,
        'bodies': bodies # Keep track of all bodies in this agglomerate to delete them later
    }

def initialize():
    create_bounding_box()
    for _ in range(NUM_SINGLE_PELLETS):
        spawn_pellet(is_agglomerate=False)
        
    for _ in range(NUM_AGGLOMERATES):
        spawn_pellet(is_agglomerate=True)

def check_despawns():
    to_remove = []
    for base_id, info in all_pellets.items():
        pos, _ = p.getBasePositionAndOrientation(base_id)
        if pos[2] < DESPAWN_Z:
            to_remove.append(base_id)
            
    for base_id in to_remove:
        info = all_pellets[base_id]
        is_agg = info['is_agglomerate']
        
        # Clean up pybullet objects
        p.removeUserDebugItem(info['text_id'])
        for b in info['bodies']:
            p.removeBody(b)
            
        del all_pellets[base_id]
        
        # Respawn with new shapes and structures
        spawn_pellet(is_agglomerate=is_agg)

def update(update_text=False):
    check_despawns()
    p.stepSimulation()
    
    # Only update debug text positions occasionally to prevent massive slowdowns
    if update_text:
        for base_id, info in all_pellets.items():
            try:
                pos, _ = p.getBasePositionAndOrientation(base_id)
                text_color = [1, 0.2, 0.2] if info['is_agglomerate'] else [0.2, 1, 0.2]
                text_pos = [pos[0], pos[1], pos[2] + BASE_RADIUS*1.5]
                info['text_id'] = p.addUserDebugText(str(info['id']), text_pos, textColorRGB=text_color, textSize=1.5, replaceItemUniqueId=info['text_id'])
            except:
                pass